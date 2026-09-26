"""Project hidden states (and re-encoded SAE features) onto the orthogonal
complement of one or more compliance directions.

Coordinate-space contract — clarifies the open issue flagged in the spec
=========================================================================

The §4.3 readout in `sae_v3_analysis/src/run_groupkfold_recompute.py` runs
Ridge on **SAE features** (sparse codes from GemmaScope/LlamaScope), not on
raw hidden states. The compliance directions, however, are derived in
**hidden-state space** (R^d_model). These two spaces are linked by the SAE
encoder:

    F = SAE_encode(h)             where h ∈ R^d_model, F ∈ R^n_sae

There are two coherent ways to "residualise":

  (A) hidden_state mode:
      project h onto the orthogonal complement of d in R^d_model, then
      re-encode through the SAE. This is the cleaner Persona-Vectors-style
      operation (the residualisation acts on the underlying representation
      that the SAE reads), at the cost of running the SAE forward pass.
          h'  = h − (h·d / ||d||²) d
          F'  = SAE_encode(h')

  (B) sae_decoder mode (cheap fallback):
      pull the direction *into the feature space* via the SAE encoder
      Jacobian and then project F directly. Algebraically this is only
      exact when the SAE is linear; for ReLU/JumpReLU SAEs it is an
      approximation. Given that GemmaScope is JumpReLU and LlamaScope is
      ReLU, mode (A) is preferred and is the default in the YAML.

This module implements mode (A) end-to-end. Mode (B) is exposed as a
faster sanity-check path (no model reload) but documented as approximate.

Inputs
------
* compliance directions: produced by `extract_compliance_directions.py`,
  stored as direction_{model}_{name}.npz (key `direction`, shape (d_model,)).
* cached hidden states: hidden_states_dp.npz (per (paradigm, model)),
  shape (n, n_layers, d_model) with `layers` array indexing the layer dim.
* (mode A only) GemmaScope or LlamaScope SAE for re-encoding.

Outputs
-------
A residualised SAE feature matrix saved as sparse COO to:
  <output.residualised_subdir>/{model}_{paradigm}_L{layer}_{mode_tag}.npz
where mode_tag is e.g. "individual_d_comp", "individual_d_agree",
"individual_d_role", "joint_3direction", or "control_random{k}".
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "m5_config.yaml"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m5.residualise")

# ----- linear algebra ------------------------------------------------------


def projection_matrix_from_direction(d: np.ndarray) -> np.ndarray:
    """Return P = d d^T / ||d||² for one direction. Shape (d_model, d_model).

    Projecting onto the *orthogonal complement* is then `(I − P) @ x`.
    """
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    norm_sq = float(d @ d)
    if norm_sq == 0:
        raise ValueError("zero-norm direction")
    return np.outer(d, d) / norm_sq


def projection_matrix_joint(directions: List[np.ndarray]) -> np.ndarray:
    """Return P = U U^T where U is an orthonormal basis of span(d_1,...,d_k).

    Built via QR of the stacked direction matrix. Robust to (near-)collinear
    directions: any column with negligible R-diagonal is dropped.
    """
    D = np.stack([np.asarray(d, dtype=np.float64).reshape(-1) for d in directions], axis=1)
    # economic QR: D = Q R, Q is (d_model, k), R is (k, k)
    Q, R = np.linalg.qr(D, mode="reduced")
    diag = np.abs(np.diag(R))
    keep = diag > 1e-8 * diag.max() if diag.max() > 0 else np.zeros_like(diag, bool)
    if not keep.any():
        raise ValueError("all directions degenerate")
    U = Q[:, keep]
    return U @ U.T


def residualise_hidden_states(H: np.ndarray, P: np.ndarray) -> np.ndarray:
    """H' = H − H @ P. H is (n, d_model); P is (d_model, d_model).

    This applies (I − P) on the right; equivalently for each row h_i,
    h_i' = h_i − P^T h_i = h_i − P h_i (P is symmetric).
    """
    if H.ndim != 2:
        raise ValueError(f"H must be (n, d_model), got {H.shape}")
    if P.shape != (H.shape[1], H.shape[1]):
        raise ValueError(f"P shape {P.shape} incompatible with H {H.shape}")
    H_dt = H.dtype
    return (H.astype(np.float64) - H.astype(np.float64) @ P).astype(H_dt)


# ----- SAE re-encoding -----------------------------------------------------


def _load_gemma_scope_sae(layer: int, sae_id_template: str, release: str, device: str):
    """Load GemmaScope (JumpReLU) via sae_lens. Lazily imported."""
    from sae_lens import SAE  # noqa: WPS433
    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id_template.format(layer=layer),
        device=device,
    )
    if isinstance(sae, tuple):  # sae_lens returns (sae, cfg, sparsity)
        sae = sae[0]
    return sae


def _load_fnlp_sae(layer: int, repo_id: str, ckpt_template: str, hp_template: str, device: str):
    """Load fnlp LlamaScope SAE (ReLU encode). Mirrors extract_llama_sm.py."""
    import math
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_template.format(layer=layer))
    hp_path = hf_hub_download(repo_id=repo_id, filename=hp_template.format(layer=layer))
    with open(hp_path) as f:
        hp = json.load(f)
    dataset_norm = hp["dataset_average_activation_norm"]["in"]
    d_model = hp["d_model"]
    norm_factor = math.sqrt(d_model) / dataset_norm
    weights = load_file(ckpt_path, device="cpu")
    W_E = weights["encoder.weight"].T.float().to(device)
    b_E = weights["encoder.bias"].float().to(device)
    return {"W_E": W_E, "b_E": b_E, "norm_factor": norm_factor, "kind": "fnlp"}


def sae_encode(H: np.ndarray, sae_obj, kind: str, device: str, batch_size: int = 256) -> np.ndarray:
    """Encode (n, d_model) → (n, n_sae) with the appropriate SAE.

    Result is dense float32 (callers convert to sparse COO before saving).
    """
    import torch

    h = torch.tensor(H, dtype=torch.float32, device=device)
    out: List[np.ndarray] = []
    for start in range(0, h.shape[0], batch_size):
        chunk = h[start : start + batch_size]
        with torch.no_grad():
            if kind == "gemma_scope":
                feats = sae_obj.encode(chunk)
            elif kind == "fnlp":
                x = chunk * sae_obj["norm_factor"]
                feats = torch.relu(x @ sae_obj["W_E"] + sae_obj["b_E"])
            else:
                raise ValueError(f"unknown SAE kind: {kind}")
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def dense_to_sparse_coo(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(F)
    vals = F[rows, cols]
    return rows.astype(np.int32), cols.astype(np.int32), vals.astype(np.float32)


# ----- IO ------------------------------------------------------------------


def load_hidden_states(cfg: dict, model: str, paradigm_dir: str) -> Dict:
    """Load `hidden_states_dp.npz` for one (model, paradigm) pair.

    Returns dict with hidden_states (n, d_model) at the configured layer plus
    the meta arrays needed downstream by the readout.
    """
    root = Path(cfg["data"]["hidden_states_root"])
    fname = cfg["data"]["hidden_states_filename"]
    path = root / paradigm_dir / model / fname
    if not path.exists():
        raise FileNotFoundError(f"missing hidden state cache: {path}")
    z = np.load(path, allow_pickle=False)
    layers = list(z["layers"])
    layer = int(cfg["layer"])
    if layer not in layers:
        raise ValueError(
            f"layer {layer} not present in {path}; cached layers={layers}"
        )
    li = layers.index(layer)
    out = {
        "H": z["hidden_states"][:, li, :].astype(np.float32),
        "valid_mask": z["valid_mask"],
        "game_ids": z["game_ids"],
        "round_nums": z["round_nums"],
        "balances": z["balances"],
        "bet_types": z["bet_types"],
        "prompt_conditions": z["prompt_conditions"],
        "game_outcomes": z["game_outcomes"],
        "layer": layer,
        "source_path": str(path),
    }
    return out


def save_residualised_sparse(
    out_path: Path,
    F_resid: np.ndarray,
    meta: Dict,
    mode_tag: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols, vals = dense_to_sparse_coo(F_resid)
    np.savez(
        out_path,
        row_indices=rows,
        col_indices=cols,
        values=vals,
        shape=np.array(F_resid.shape, dtype=np.int64),
        # meta arrays (mirror sae_features_L*.npz layout for downstream parity)
        game_ids=meta["game_ids"],
        round_nums=meta["round_nums"],
        balances=meta["balances"],
        bet_types=meta["bet_types"],
        prompt_conditions=meta["prompt_conditions"],
        game_outcomes=meta["game_outcomes"],
    )
    side = out_path.with_suffix(".json")
    with open(side, "w") as f:
        json.dump(
            {
                "mode_tag": mode_tag,
                "n_rows": int(F_resid.shape[0]),
                "n_features": int(F_resid.shape[1]),
                "nnz": int(len(vals)),
                "layer": meta["layer"],
                "source_hidden_states": meta["source_path"],
            },
            f,
            indent=2,
        )
    log.info("saved residualised features to %s (nnz=%d)", out_path, len(vals))


# ----- driver --------------------------------------------------------------


def run_residualisation(
    cfg: dict,
    model: str,
    paradigm_dir: str,
    directions_dir: Path,
    out_dir: Path,
    modes: List[str],
    n_random_baseline: int = 0,
) -> Dict:
    """Apply each requested residualisation mode to one (model, paradigm).

    Returns a summary dict mapping mode_tag → output_path.
    """
    layer = int(cfg["layer"])
    space = cfg.get("residualisation_space", "hidden_state")
    if space != "hidden_state":
        raise NotImplementedError(
            f"residualisation_space='{space}' not implemented; "
            "current impl supports 'hidden_state' only — see module docstring",
        )

    # Load cached hidden states + meta.
    hs = load_hidden_states(cfg, model, paradigm_dir)
    H = hs["H"]
    n, d_model = H.shape
    log.info("loaded H[%s/%s] shape=%s", model, paradigm_dir, H.shape)

    # Load direction vectors per direction.
    direction_arrays: Dict[str, np.ndarray] = {}
    for dname in cfg["probe_directions"].keys():
        p = directions_dir / f"direction_{model}_{dname}.npz"
        if not p.exists():
            raise FileNotFoundError(f"missing direction file: {p}")
        z = np.load(p, allow_pickle=False)
        d = z["direction"]
        if d.shape[0] != d_model:
            raise ValueError(
                f"direction d_model={d.shape[0]} disagrees with H d_model={d_model}",
            )
        direction_arrays[dname] = d

    # Build per-mode projection matrices.
    proj_specs: List[Tuple[str, np.ndarray]] = []
    if "individual_per_direction" in modes:
        for dname, d in direction_arrays.items():
            proj_specs.append((f"individual_{dname}", projection_matrix_from_direction(d)))
    if "joint_3direction" in modes:
        proj_specs.append(
            ("joint_3direction", projection_matrix_joint(list(direction_arrays.values())))
        )
    # Optional random-direction control(s).
    rng = np.random.default_rng(int(cfg.get("seeds", {}).get("numpy", 42)))
    for k in range(n_random_baseline):
        rd = rng.standard_normal(d_model).astype(np.float64)
        proj_specs.append((f"control_random{k}", projection_matrix_from_direction(rd)))

    # SAE setup (mode A) — load once and reuse for all projections.
    sae_obj, sae_kind = _load_sae_for_model(cfg, model, layer)

    summary: Dict[str, str] = {}
    for mode_tag, P in proj_specs:
        log.info("[%s/%s] residualising mode=%s", model, paradigm_dir, mode_tag)
        H_res = residualise_hidden_states(H, P)
        F_res = sae_encode(
            H_res,
            sae_obj,
            sae_kind,
            cfg.get("generation", {}).get("device", "cuda"),
        )
        out_path = out_dir / f"{model}_{paradigm_dir}_L{layer}_{mode_tag}.npz"
        save_residualised_sparse(out_path, F_res, hs, mode_tag)
        summary[mode_tag] = str(out_path)
    return summary


def _load_sae_for_model(cfg: dict, model: str, layer: int):
    mc = cfg["models"][model]
    device = cfg.get("generation", {}).get("device", "cuda")
    if mc["sae_kind"] == "gemma_scope":
        sae = _load_gemma_scope_sae(layer, mc["sae_id_template"], mc["sae_release"], device)
        return sae, "gemma_scope"
    if mc["sae_kind"] == "fnlp":
        sae = _load_fnlp_sae(
            layer,
            mc["sae_repo_id"],
            mc["sae_filename_template"],
            mc["sae_hp_template"],
            device,
        )
        return sae, "fnlp"
    raise ValueError(f"unknown sae_kind for model {model}: {mc['sae_kind']}")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M5 — residualise SAE features against compliance dirs")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", choices=["gemma", "llama", "all"], default="all")
    p.add_argument(
        "--paradigm-dir", default="slot_machine",
        help="subdirectory under data.hidden_states_root (matches sae_v3_analysis layout)",
    )
    p.add_argument("--directions-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--n-random-baseline", type=int, default=None)
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_root = Path(args.output or cfg["output"]["root"])
    directions_dir = args.directions_dir or (
        out_root / cfg["output"].get("directions_subdir", "directions")
    )
    resid_dir = out_root / cfg["output"].get("residualised_subdir", "residualised")
    resid_dir.mkdir(parents=True, exist_ok=True)

    n_random = args.n_random_baseline
    if n_random is None:
        controls = cfg.get("controls", {})
        n_random = int(controls.get("n_random_baseline", 0)) if controls.get("random_direction_baseline") else 0

    models = ["gemma", "llama"] if args.model == "all" else [args.model]
    summary: Dict[str, Dict[str, str]] = {}
    for mk in models:
        summary[mk] = run_residualisation(
            cfg, mk, args.paradigm_dir, Path(directions_dir), resid_dir,
            modes=cfg.get("modes", ["individual_per_direction", "joint_3direction"]),
            n_random_baseline=n_random,
        )

    out_summary = resid_dir / f"residualisation_summary_{args.paradigm_dir}.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote %s", out_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
