"""Trial loop: one arm = N single-decision trials (M3'' protocol, multi-layer)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime

import numpy as np

from . import wandb_logger
from .checkpoint import ArmCheckpoint, VectorStore
from .paper_axes import MODEL_DIMS
from .hooks import (HookGroup, MultiLayerPatcher, MultiLayerProjector,
                    MultiLayerSteerer, PrefillTap, SubspacePatcher,
                    cache_layer_outputs)
from .prompts import build_prompt, parse_response, twin_combo
from .states import ensure_sm_catalog, load_minusG_states

# Model ids mirror paper_experiments/slot_machine_6models/src/
# llama_gemma_experiment.py::load_model exactly (the §3 behavioural corpus).
MODEL_PATHS = {
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}
MODEL_PATH = MODEL_PATHS["gemma"]  # frozen alias (gemma asset/test surface)
# Gemma asset-schema constants (directions npz rows, VectorStore stacks built
# so far), resolved from the single MODEL_DIMS registry so gemma stays 42/3584.
# The runner path itself derives n_layers/d_model from the loaded model config
# (_model_dims) — do NOT use these for anything model-dependent.
N_LAYERS, D_MODEL = MODEL_DIMS["gemma"]
TEMPERATURE, MAX_NEW_TOKENS = 0.7, 200
SEED_BASE = 42
PROBE_MODES = {"anchor_minus", "anchor_plus", "patch", "steer"}
IC_MODES = {"anchor_minus", "steer"}
MW_MODES = {"anchor_minus", "steer"}


def assert_replay_match(arm_hash, dir_hash):
    """Replay guard (past-failure fix: V12/V14 headline invalid because the
    direction was built on one prompt set and tested on another). The arm's
    replay prompt-set hash MUST equal the direction-build prompt-set hash;
    raise ValueError on mismatch so a mis-paired arm fails loudly, never
    silently steers the wrong prompts."""
    if str(arm_hash) != str(dir_hash):
        raise ValueError(
            f"replay-set mismatch: arm prompt_set_hash {arm_hash!r} != "
            f"direction build hash {dir_hash!r} — refusing to replay")


def _replay_hash(arm):
    """Deterministic hash of an arm's replay-relevant config (prompt set,
    direction asset, layers, alpha) — recorded in the run-metadata summary and
    usable by assert_replay_match."""
    key = {
        "prompt_set": arm.get("prompt_set"),
        "directions_npz": arm.get("directions_npz"),
        "layers": arm.get("layers"),
        "alpha": arm.get("alpha"),
        "mode": arm.get("mode"),
        "task": arm.get("task", "sm"),
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def sec4_prompt_set_hash(prompt_set):
    """Canonical hash of a replay prompt-set label. The axis builder stamps the
    build-time label into the npz (indicator_axes.BUILD_PROMPT_SET); the runner
    hashes the arm's own prompt_set with this SAME function and compares — a
    mis-paired arm (direction built on one prompt set, replayed on another) then
    fails assert_replay_match loudly instead of silently steering the wrong
    distribution (the V12/V14 headline-invalidation failure)."""
    return hashlib.sha1(str(prompt_set).encode()).hexdigest()[:16]


def _sec4_provenance(arm):
    """Build provenance recorded in the arm's steering npz by indicator_axes, or
    None for arms that carry no built sec4 direction — random-null / anchor
    baselines (no eligible npz) and prior-wave assets without the sec4 keys
    (kept exempt so prior waves stay byte-identical)."""
    npz = arm.get("directions_npz")
    if arm.get("mode") != "steer" or arm.get("direction") == "random" or not npz:
        return None
    z = np.load(npz)
    if "build_prompt_set" not in z.files:
        return None
    ids = (set(int(g) for g in z["build_game_ids"])
           if "build_game_ids" in z.files else None)
    return {"prompt_set": str(z["build_prompt_set"]), "build_game_ids": ids,
            # build task (indicator_axes stamps it): the IC replay guard only
            # compares ids within the SAME id space (IC-built npz vs IC pool).
            "task": str(z["task"]) if "task" in z.files else None}


def _replay_game_ids(model, n, offset, pool_len):
    """1-based catalog game ids of the −G offset-slice this SM arm replays, or
    None when the corpus is not a single catalog file (id reconstruction needs
    the exact load_minusG_states iteration order). Reuses the frozen
    axes.minusG_pool_with_game_ids pool (identical Random(42) shuffle)."""
    from .axes import minusG_pool_with_game_ids
    files = sorted(ensure_sm_catalog(model).glob("*.json"))
    if len(files) != 1:
        return None
    pool = minusG_pool_with_game_ids(str(files[0]))
    if not pool:
        return None
    return {pool[(i + offset) % pool_len][0] for i in range(n)}


def _assert_replay_ok(arm, model, n, offset, pool_len, twin, ic_replay_ids=None):
    """Replay guard (past-failure V12/V14): assert the arm replays the SAME
    prompt set the axis was built on AND, for the SM −G pool, a game slice
    DISJOINT from the axis build set. No-op for null / baseline / prior-wave
    arms (npz without sec4 build provenance).

    ic_replay_ids (sec4_w3, additive): the IC game ids the arm will replay,
    passed by _run_arm_ic from its loaded pool. Checked against the npz build
    ids ONLY when the npz was built on the IC corpus (stamped task
    'investment_choice' — same global-counter id space); SM-/shared3-stamped
    npz carry SM catalog ids, so comparing them to IC counters would be a
    cross-id-space false alarm."""
    prov = _sec4_provenance(arm)
    if prov is None:
        return
    assert_replay_match(sec4_prompt_set_hash(arm.get("prompt_set")),
                        sec4_prompt_set_hash(prov["prompt_set"]))
    if prov["build_game_ids"] and twin == "G":
        replay_ids = _replay_game_ids(model, n, offset, pool_len)
        if replay_ids is not None:
            overlap = prov["build_game_ids"] & replay_ids
            assert not overlap, (
                f"{arm['id']}: {len(overlap)} replayed games leak into the axis "
                f"build set — held-out disjointness violated")
    if (prov["build_game_ids"] and ic_replay_ids is not None
            and prov["task"] == "investment_choice"):
        overlap = prov["build_game_ids"] & set(ic_replay_ids)
        assert not overlap, (
            f"{arm['id']}: {len(overlap)} replayed IC games leak into the axis "
            f"build set — held-out disjointness violated")


def _arm_metrics(rows):
    """Per-arm behaviour + manipulation-check summary from the recorded rows.

    manip_proj reads rec['vector_log']['proj'] (key-path fix: the manip check
    was NaN everywhere because it read a non-existent top-level key)."""
    parsed = [r for r in rows if r.get("parse_ok")]
    br = [float(r["bet_ratio"]) for r in parsed if r.get("bet_ratio") is not None]
    risky = [1.0 if r.get("risky") else 0.0 for r in parsed if "risky" in r]
    projs = [float(r["vector_log"]["proj"]) for r in rows
             if isinstance(r.get("vector_log"), dict)
             and r["vector_log"].get("proj") is not None]
    m = {
        "n": len(rows),
        "parse_rate": (len(parsed) / len(rows)) if rows else float("nan"),
    }
    if br:
        m["mean_bet_ratio"] = float(np.mean(br))
    if risky:
        m["risky_rate"] = float(np.mean(risky))
    if projs:
        m["mean_manip_proj"] = float(np.mean(projs))
    return m


def _arm_done(arm, ck, t_start):
    """Called at every arm DONE: log the per-arm summary to W&B (no-op unless
    enabled) and write the run-metadata summary (n, t_start, t_end,
    replay_hash) — provenance guard against smoke JSONs read as real runs."""
    rows = [json.loads(l) for l in open(ck.path)] if ck.path.exists() else []
    metrics = _arm_metrics(rows)
    wandb_logger.log_arm(arm["id"], arm.get("alpha"), metrics)
    ck.record_summary({
        "arm": arm["id"], "phase": arm.get("phase"),
        "n": len(rows), "t_start": t_start, "t_end": time.time(),
        "replay_hash": _replay_hash(arm),
        "prompt_set": arm.get("prompt_set"),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    })


def load_model(gpu: int, model_name: str = "gemma"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = MODEL_PATHS[model_name]
    tok = AutoTokenizer.from_pretrained(path, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{gpu}"}, token=os.environ.get("HF_TOKEN"))
    model.eval()
    # Model-aware geometry guard via the single MODEL_DIMS registry: gemma keeps
    # its frozen W1/W2 42/3584 contract byte-identically; llama additionally
    # asserts 32/4096 (W8 model symmetry) instead of skipping the check.
    n_exp, d_exp = MODEL_DIMS[model_name]
    assert len(model.model.layers) == n_exp, \
        f"unexpected layer count for {model_name}"
    assert model.config.hidden_size == d_exp, \
        f"unexpected hidden size for {model_name}"
    return model, tok, f"cuda:{gpu}"


def _load_model_for(arm, gpu):
    """Gemma arms keep the frozen single-arg call shape (dry-run tests
    monkeypatch load_model as ``lambda gpu: ...``); llama arms pass the name."""
    name = arm.get("model", "gemma")
    return load_model(gpu) if name == "gemma" else load_model(gpu, name)


def _model_dims(model):
    """(n_layers, d_model) from the LOADED model — never from constants."""
    return len(model.model.layers), int(model.config.hidden_size)


def _generate(model, tok, device, prompt, hookset, seed):
    import torch
    torch.manual_seed(seed)
    inputs = tok(prompt, return_tensors="pt").to(device)
    if hookset is not None:
        hookset.install(model)
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=TEMPERATURE, do_sample=True,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)
    finally:
        if hookset is not None:
            hookset.remove()


def _load_steer_assets(arm, d_model=D_MODEL):
    """directions npz: keys 'directions' (L,D), 'scales' (L,) — built by analyze.py.

    direction: random → isotropic unit vectors (control), same per-layer scales.
    d_model only sizes the random control draw; npz rows carry their own dim.

    W3 assets (paper_axes.py) record gate_passed inside the npz; a failed
    reproduction gate auto-excludes the arm (RUN_PLAN_W3.md 구현 원칙), enforced
    here at load time. W2-era npz without the field stay exempt; random-direction
    arms only borrow per-layer scales, so the gate does not apply to them.
    """
    import torch
    z = np.load(arm["directions_npz"])
    scales = {li: float(z["scales"][li]) for li in arm["layers"]}
    if arm.get("direction") == "random":
        rng = np.random.Generator(np.random.PCG64(int(arm["dir_seed"])))
        dirs = {}
        for li in arm["layers"]:
            v = rng.standard_normal(d_model)
            dirs[li] = torch.tensor(v / np.linalg.norm(v), dtype=torch.float32)
        return dirs, scales
    if "gate_passed" in z.files:
        assert bool(z["gate_passed"]), (
            f"{arm['id']}: direction asset {arm['directions_npz']} failed its "
            f"reproduction gate — arm auto-excluded (RUN_PLAN_W3.md)")
    dirs = {li: torch.tensor(z["directions"][li], dtype=torch.float32)
            for li in arm["layers"]}
    return dirs, scales


def _load_projectout_dirs(arm, d_model=D_MODEL):
    """W13 NECESSITY: per-layer axis rows for project-out (MultiLayerProjector).

    Reuses the SAME directions npz the steerer reads (keys 'directions' (L,D)),
    applying the identical reproduction gate; the projector unit-normalises each
    row itself, so no scales are needed (removal is a pure geometric projection,
    not a sigma-unit dose). direction: random draws isotropic unit rows (the
    project-out null control) from the arm's dir_seed — removing a RANDOM
    direction should NOT lower betting, the necessity contrast. Only d_model
    sizes the random draw; real rows carry their own dim."""
    import torch
    z = np.load(arm["directions_npz"])
    if arm.get("direction") == "random":
        rng = np.random.Generator(np.random.PCG64(int(arm["dir_seed"])))
        dirs = {}
        for li in arm["layers"]:
            v = rng.standard_normal(d_model)
            dirs[li] = torch.tensor(v / np.linalg.norm(v), dtype=torch.float32)
        return dirs
    if "gate_passed" in z.files:
        assert bool(z["gate_passed"]), (
            f"{arm['id']}: direction asset {arm['directions_npz']} failed its "
            f"reproduction gate — arm auto-excluded (RUN_PLAN_W3.md)")
    return {li: torch.tensor(z["directions"][li], dtype=torch.float32)
            for li in arm["layers"]}


def _load_bases(arm, d_model=D_MODEL):
    import torch
    if arm.get("basis", "pca") == "random":
        rng = np.random.Generator(np.random.PCG64(int(arm["basis_seed"])))
        bases = {}
        for li in arm["layers"]:
            g = rng.standard_normal((d_model, int(arm["r"])))
            q, _ = np.linalg.qr(g)
            bases[li] = torch.tensor(q, dtype=torch.float32)
        return bases
    z = np.load(arm["basis_npz"])
    return {li: torch.tensor(z[f"L{li}"][:, :int(arm["r"])], dtype=torch.float32)
            for li in arm["layers"]}


def _tap_projection(arm, tap):
    """Manipulation-check entry for steer arms with log_vectors (w3bk spec).

    Projection of the prefill last-token hidden state at the tap layer
    (default L22, arm key 'log_layer') onto that layer's UNIT direction row
    from the arm's npz. The raw (d_model,) vector is too large for per-trial
    jsonl; the scalar projection answers the check — did the reading at the
    monitored layer move along the steered axis.
    """
    li = tap.layer_indices[0]
    z = np.load(arm["directions_npz"])
    d = np.asarray(z["directions"][li], dtype=np.float64)
    d /= max(float(np.linalg.norm(d)), 1e-12)
    h = tap.hidden.numpy().astype(np.float64)
    return {"layer": li, "proj": float(h @ d),
            "h_norm": float(np.linalg.norm(h))}


def build_branch_game(game, round_idx, bet, win: bool) -> dict:
    """Counterfactual NEXT-round game state after betting `bet` at round_idx.

    Pure: returns a NEW dict, never mutates `game`. WIN pays 3.0x the bet
    (net +2*bet), LOSS forfeits the bet — same arithmetic as the §3 engine.
    The appended history entry renders in build_prompt as
    'Round {round_idx+1}: Bet ${bet}, WIN|LOSS, Balance ${new_bal}'.
    All-in bets are rejected: a LOSS branch would be a probe-bankruptcy
    (balance 0, no next decision) — callers log those instead of branching.
    """
    decs = game["decisions"]
    assert 0 <= round_idx < len(decs), f"round_idx {round_idx} out of range"
    bal = decs[round_idx]["balance_before"]
    assert 0 < bet < bal, f"branch needs 0 < bet < balance (bet={bet}, bal={bal})"
    new_bal = bal + 2 * bet if win else bal - bet
    return {
        "bet_type": game["bet_type"],
        "prompt_combo": game.get("prompt_combo", ""),
        "decisions": list(decs[:round_idx + 1]) + [{"balance_before": new_bal}],
        "history": list(game.get("history", [])[:round_idx])
        + [{"round": round_idx + 1, "bet": bet, "win": win, "balance": new_bal}],
    }


def _run_probe(arm, model, tok, device, game, round_idx, action, amount, balance,
               seed, branch_combo, plus_combo, d_model=D_MODEL):
    """Stage-2 LC probe: LOSS branch (seed+1) + WIN branch (seed+2).

    branch_combo = combo of the EVALUATED stage-1 prompt (plus combo for
    anchor_plus arms, minus combo otherwise) — branches see the same framing.
    """
    if action == "stop":
        return {"stage1_stop": True}
    if amount >= balance:
        return {"all_in": True}  # probe-bankruptcy on LOSS; LC undefined, no branches
    r_t = amount / balance
    steer_assets = _load_steer_assets(arm, d_model) if arm["mode"] == "steer" else None
    out = {"r_t": r_t, "bet": amount}
    for key, win, seed_off in (("loss", False, 1), ("win", True, 2)):
        bg = build_branch_game(game, round_idx, amount, win)
        eval_p = build_prompt(bg, round_idx + 1, override_combo=branch_combo)
        assert eval_p is not None, "branch prompt build failed"
        hookset = None
        if arm["mode"] == "patch":
            # fresh donor-twin cache OF THE BRANCH prompt (plus_combo carries
            # the arm's twin_component recipe from run_arm; +G unless set)
            plus_p = build_prompt(bg, round_idx + 1, override_combo=plus_combo)
            cached, _ = cache_layer_outputs(model, tok, device, plus_p, arm["layers"])
            hookset = MultiLayerPatcher(cached)
        elif arm["mode"] == "steer":
            dirs, scales = steer_assets
            hookset = MultiLayerSteerer(dirs, scales, alpha=float(arm["alpha"]))
        text = _generate(model, tok, device, eval_p, hookset, seed + seed_off)
        new_bal = bg["decisions"][-1]["balance_before"]
        b_action, b_amount = parse_response(text, min(100, new_bal))
        r = b_amount / new_bal if b_action == "bet" else 0.0
        lc = max(0.0, (r - r_t) / r_t) if b_action == "bet" else None
        out[key] = {
            "action": b_action, "amount": b_amount, "r": r, "lc": lc,
            "parse_ok": bool(re.search(r"Final Decision:", text, re.IGNORECASE)),
            "response": text[:300],
        }
    return out


def run_arm(arm, out_dir, gpu=0, n=None, smoke=False):
    if arm.get("task", "sm") == "ic":
        return _run_arm_ic(arm, out_dir, gpu=gpu, n=n, smoke=smoke)
    if arm.get("task", "sm") == "mw":
        return _run_arm_mw(arm, out_dir, gpu=gpu, n=n, smoke=smoke)
    probe = bool(arm.get("probe"))
    if probe:
        assert arm.get("task", "sm") != "ic", "probe arms are SM-only"
        assert arm["mode"] in PROBE_MODES, f"probe unsupported for mode {arm['mode']}"
    n = n or arm["n"]
    if smoke:
        n = min(n, 3)
    phase = arm["phase"]
    ck = ArmCheckpoint(phase, arm["id"], out_dir,
                       hf_enabled=False if smoke else None)
    _t_start = time.time()
    ensure_sm_catalog(arm["model"])
    # Confirmatory arms use a HELD-OUT slice of the frozen pool (state_offset
    # past the discovery indices) and a distinct seed_base.
    offset = int(arm.get("state_offset", 0))
    seed_base = int(arm.get("seed_base", SEED_BASE))
    twin = arm.get("twin_component", "G")
    # twin_combo is idempotent when the base combo already contains the twin
    # component (donor combo == base combo → identity patch, the trial measures
    # nothing). The frozen −G pool only filters G, so for non-G twins (w3m +M)
    # restrict the eval slice to twin-free base combos: scan FORWARD from
    # state_offset in frozen pool order, keeping the slice inside the same
    # held-out window the W2 anchors sampled (RUN_PLAN_W3.md w3m — anchor
    # comparison is restricted to the matching twin-free stratum at analysis).
    pad = 50 if twin == "G" else 3 * n + 50
    states = load_minusG_states(arm["model"], n=n + offset + pad)
    assert states, "empty state pool"
    if twin != "G":
        states = [s for s in states[offset:]
                  if twin not in s[0].get("prompt_combo", "")]
        assert len(states) >= n, \
            f"{arm['id']}: twin-free state pool too small ({len(states)} < {n})"
        offset = 0
    _assert_replay_ok(arm, arm["model"], n, offset, len(states), twin)
    state_filter = arm.get("state_filter")
    if state_filter:
        # sec4_w3 Q1 rung-2 (additive; no-op when absent): condition the replay
        # on the PREVIOUS round's outcome. The candidate window is EXACTLY the
        # n states the unfiltered arm would replay (same offset slice, so the
        # axis-build disjointness just asserted covers every filtered state);
        # trials wrap around the postloss/postwin partition of it with the
        # alpha-independent seeds intact. Labeling is postloss_analysis.
        # label_postloss — the SAME function the rung-1 re-analysis uses.
        from .postloss_analysis import label_postloss
        want_loss = state_filter == "postloss"
        window = [states[(i + offset) % len(states)] for i in range(n)]
        states = [s for s in window if label_postloss(s) is want_loss]
        offset = 0
        assert states, f"{arm['id']}: state_filter={state_filter} left an empty pool"
        print(f"[{arm['id']}] state_filter={state_filter} pool={len(states)} "
              f"of window {len(window)} (wrap-around allowed)", flush=True)
    model, tok, device = _load_model_for(arm, gpu)
    n_layers, d_model = _model_dims(model)
    vec = None
    if arm.get("log_vectors") and arm["mode"] != "steer":
        # patch arms: full −G/+G last-token stacks (e1_full protocol).
        # steer arms log a per-trial projection instead (see _tap_projection).
        vec = VectorStore(ck.path.with_name(f"{arm['id']}_vectors.npz"),
                          n_layers, d_model)
    done = ck.done_seeds()
    print(f"[{arm['id']}] n={n} done={len(done)} pool={len(states)} "
          f"offset={offset}", flush=True)

    for i in range(n):
        seed = seed_base + i * 997
        if seed in done:
            continue
        t0 = time.time()
        game, round_idx = states[(i + offset) % len(states)]
        base_combo = game.get("prompt_combo", "")
        # Donor twin combo: +G by default (frozen W1/W2 recipe, byte-identical
        # via twin_combo), +M for w3m mediation arms (twin_component: M).
        # _run_probe re-derives its branch donor from this same plus_combo.
        plus_combo = twin_combo(base_combo, arm.get("twin_component", "G"))
        minus_p = build_prompt(game, round_idx, override_combo=base_combo)
        plus_p = build_prompt(game, round_idx, override_combo=plus_combo)
        if minus_p is None or plus_p is None:
            continue

        mode = arm["mode"]
        eval_p, hookset, tap = minus_p, None, None
        if mode == "anchor_plus":
            eval_p = plus_p
        elif mode == "patch":
            cached, _ = cache_layer_outputs(model, tok, device, plus_p, arm["layers"])
            hookset = MultiLayerPatcher(cached)
            if vec is not None and seed not in vec.seeds:
                full = list(range(n_layers))
                cached_m, _ = cache_layer_outputs(model, tok, device, minus_p, full)
                cached_p, _ = cache_layer_outputs(model, tok, device, plus_p, full)
                vec.append(seed,
                           np.stack([cached_m[l][-1].float().cpu().numpy()
                                     for l in full]),
                           np.stack([cached_p[l][-1].float().cpu().numpy()
                                     for l in full]))
        elif mode == "subspace":
            cached, _ = cache_layer_outputs(model, tok, device, plus_p, arm["layers"])
            hookset = SubspacePatcher(cached, _load_bases(arm, d_model))
        elif mode == "steer":
            dirs, scales = _load_steer_assets(arm, d_model)
            hookset = MultiLayerSteerer(dirs, scales, alpha=float(arm["alpha"]))
            if arm.get("twin"):
                # sec4_w3 Q3 (additive; no-op when absent): steer the +twin
                # prompt — the dose-response measured UNDER the +G (or +M)
                # condition, built through the same frozen twin_combo recipe
                # the patch arms' donors use.
                eval_p = build_prompt(game, round_idx,
                                      override_combo=twin_combo(base_combo,
                                                                arm["twin"]))
                assert eval_p is not None, "twin eval prompt build failed"
            if arm.get("log_vectors"):
                li = int(arm.get("log_layer", 22))
                assert 0 <= li < n_layers, f"log_layer {li} out of range"
                tap = PrefillTap(li)
                hookset = HookGroup(hookset, tap)  # tap last → post-edit view
        elif mode == "project_out":
            # W13 NECESSITY: REMOVE the axis from the residual stream at every
            # write-window layer, all positions, every forward (opposite of the
            # additive steerer). No alpha; removal_frac (default 1.0) scales the
            # removal h - frac*(h·û)û. Evaluates the natural −G prompt (minus_p).
            dirs = _load_projectout_dirs(arm, d_model)
            frac = float(arm.get("removal_frac", 1.0))
            hookset = MultiLayerProjector(dirs, frac=frac)
            if arm.get("log_vectors"):
                li = int(arm.get("log_layer", 22))
                assert 0 <= li < n_layers, f"log_layer {li} out of range"
                tap = PrefillTap(li)
                hookset = HookGroup(hookset, tap)  # tap last → post-removal view
        elif mode != "anchor_minus":
            raise ValueError(f"unknown mode {mode}")

        try:
            text = _generate(model, tok, device, eval_p, hookset, seed)
        except Exception as e:
            print(f"[{arm['id']} trial {i}] ERROR {type(e).__name__}: {e}", flush=True)
            continue

        bal_m = re.search(r"Current balance:\s*\$(\d+)", eval_p)
        balance = int(bal_m.group(1)) if bal_m else 100
        max_bet = min(100, balance)
        action, amount = parse_response(text, max_bet)
        rec = {
            "trial_id": i, "seed": seed, "arm": arm["id"], "phase": phase,
            "mode": mode, "layers": arm.get("layers"),
            "twin_component": arm.get("twin_component", "G"),
            "r": arm.get("r"), "alpha": arm.get("alpha"),
            "source_state": {"prompt_combo": base_combo,
                             "bet_type": game.get("bet_type"),
                             "round_idx": round_idx, "balance": balance},
            "action": action, "amount": amount,
            "bet_ratio": amount / max(balance, 1) if action == "bet" else 0.0,
            "extreme": bool(action == "bet" and amount >= max_bet),
            "parse_ok": bool(re.search(r"Final Decision:", text, re.IGNORECASE)),
            "response": text[:300], "elapsed_s": round(time.time() - t0, 1),
            "timestamp": datetime.now().isoformat(),
        }
        if arm.get("twin"):  # mark +twin-steered rows (key absent elsewhere)
            rec["twin"] = arm["twin"]
        if tap is not None and tap.hidden is not None:
            rec["vector_log"] = _tap_projection(arm, tap)
        if probe:
            branch_combo = plus_combo if mode == "anchor_plus" else base_combo
            try:
                rec["probe"] = _run_probe(arm, model, tok, device, game, round_idx,
                                          action, amount, balance, seed,
                                          branch_combo, plus_combo, d_model)
            except Exception as e:
                print(f"[{arm['id']} trial {i}] PROBE ERROR "
                      f"{type(e).__name__}: {e}", flush=True)
                continue  # skip record → resume retries the whole trial
            rec["elapsed_s"] = round(time.time() - t0, 1)
        ck.record(rec)
        if vec is not None:
            vec.save()
        print(f"  [{arm['id']} {i + 1}/{n}] {action} ${amount} "
              f"({time.time() - t0:.1f}s)", flush=True)
    ck.final_sync()
    _arm_done(arm, ck, _t_start)
    print(f"[{arm['id']}] DONE {len(ck.done_seeds())} trials", flush=True)


def _run_arm_ic(arm, out_dir, gpu=0, n=None, smoke=False):
    """Investment-choice single-decision arm (W1 §4.2 transfer bundle).

    Stored IC prompt verbatim — no minus/plus twin exists, so the only legal
    modes are anchor_minus (natural prompt, no hook) and steer.
    """
    from . import ic  # lazy: only task: ic arms need (or have) the ic module
    assert not arm.get("probe"), "probe arms are SM-only"
    mode = arm["mode"]
    assert mode in IC_MODES, f"ic: unsupported mode {mode}"
    n = n or arm["n"]
    if smoke:
        n = min(n, 3)
    phase = arm["phase"]
    ck = ArmCheckpoint(phase, arm["id"], out_dir,
                       hf_enabled=False if smoke else None)
    _t_start = time.time()
    ic.ensure_ic_catalog(arm["model"])
    offset = int(arm.get("state_offset", 0))
    seed_base = int(arm.get("seed_base", SEED_BASE))
    pool = ic.load_ic_states(arm["model"], n=n + offset + 50)
    assert pool, "empty IC state pool"
    assert len(pool) >= n, (
        f"{arm['id']}: IC pool {len(pool)} < n {n} — would silently wrap states")
    _assert_replay_ok(arm, arm["model"], n, offset, len(pool), None,
                      ic_replay_ids={int(pool[(i + offset) % len(pool)][0])
                                     for i in range(n)})
    model, tok, device = _load_model_for(arm, gpu)
    n_layers, d_model = _model_dims(model)
    done = ck.done_seeds()
    print(f"[{arm['id']}] task=ic n={n} done={len(done)} pool={len(pool)} "
          f"offset={offset}", flush=True)

    for i in range(n):
        seed = seed_base + i * 997
        if seed in done:
            continue
        t0 = time.time()
        game_id, prompt, _meta = pool[(i + offset) % len(pool)]
        assert isinstance(prompt, str) and prompt, "IC state has empty prompt"
        hookset, tap = None, None
        if mode == "steer":
            dirs, scales = _load_steer_assets(arm, d_model)
            hookset = MultiLayerSteerer(dirs, scales, alpha=float(arm["alpha"]))
            if arm.get("log_vectors"):
                li = int(arm.get("log_layer", 22))
                assert 0 <= li < n_layers, f"log_layer {li} out of range"
                tap = PrefillTap(li)
                hookset = HookGroup(hookset, tap)
        try:
            text = _generate(model, tok, device, prompt, hookset, seed)
        except Exception as e:
            print(f"[{arm['id']} trial {i}] ERROR {type(e).__name__}: {e}", flush=True)
            continue
        choice, parse_ok, risky = ic.parse_ic_response(text)
        rec = {
            "trial_id": i, "seed": seed, "arm": arm["id"], "phase": phase,
            "mode": mode, "task": "ic", "layers": arm.get("layers"),
            "alpha": arm.get("alpha"), "game_id": int(game_id),
            "choice": choice, "risky": bool(risky), "parse_ok": bool(parse_ok),
            "response": text[:300],
            "elapsed_s": round(time.time() - t0, 1),
            "timestamp": datetime.now().isoformat(),
        }
        if tap is not None and tap.hidden is not None:
            rec["vector_log"] = _tap_projection(arm, tap)
        ck.record(rec)
        print(f"  [{arm['id']} {i + 1}/{n}] choice={choice} "
              f"({time.time() - t0:.1f}s)", flush=True)
    ck.final_sync()
    _arm_done(arm, ck, _t_start)
    print(f"[{arm['id']}] DONE {len(ck.done_seeds())} trials", flush=True)


def _run_arm_mw(arm, out_dir, gpu=0, n=None, smoke=False):
    """Mystery-wheel single-decision arm (xtask 3x3 BK transfer, mw target).

    Near-verbatim copy of _run_arm_ic: the stored MW prompt is replayed verbatim
    (no minus/plus twin), so the legal modes are anchor_minus and steer. MW
    choice encoding is INVERTED vs slot machine — game choice 2 = Spin (risky),
    1 = Stop (safe) — so risky = (action == 'spin'). Balance is taken from the
    pool tuple's meta['balance_before'] rather than re-regexing the prompt
    (MW prompts say 'Current Balance:' with a capital B, unlike the SM regex);
    the variable parser needs it to compute bet_ratio.
    """
    from . import mw  # lazy: only task: mw arms need (or have) the mw module
    assert not arm.get("probe"), "probe arms are SM-only"
    mode = arm["mode"]
    assert mode in MW_MODES, f"mw: unsupported mode {mode}"
    n = n or arm["n"]
    if smoke:
        n = min(n, 3)
    phase = arm["phase"]
    ck = ArmCheckpoint(phase, arm["id"], out_dir,
                       hf_enabled=False if smoke else None)
    _t_start = time.time()
    mw.ensure_mw_catalog(arm["model"])
    offset = int(arm.get("state_offset", 0))
    seed_base = int(arm.get("seed_base", SEED_BASE))
    pool = mw.load_mw_states(arm["model"], n=n + offset + 50)
    assert pool, "empty MW state pool"
    assert len(pool) >= n, (
        f"{arm['id']}: MW pool {len(pool)} < n {n} "
        f"(variable-filtered corpus too small) — would silently wrap states")
    _assert_replay_ok(arm, arm["model"], n, offset, len(pool), None)
    model, tok, device = _load_model_for(arm, gpu)
    n_layers, d_model = _model_dims(model)
    done = ck.done_seeds()
    print(f"[{arm['id']}] task=mw n={n} done={len(done)} pool={len(pool)} "
          f"offset={offset}", flush=True)

    for i in range(n):
        seed = seed_base + i * 997
        if seed in done:
            continue
        t0 = time.time()
        game_id, prompt, meta = pool[(i + offset) % len(pool)]
        assert isinstance(prompt, str) and prompt, "MW state has empty prompt"
        # Balance from the pool tuple (MW header is capital-B 'Current Balance:',
        # incompatible with the SM regex); the variable parser needs it to clamp
        # the bet and compute bet_ratio. Falls back to 100 if the corpus omits it.
        balance = int(meta.get("balance_before") or 100)
        hookset, tap = None, None
        if mode == "steer":
            dirs, scales = _load_steer_assets(arm, d_model)
            hookset = MultiLayerSteerer(dirs, scales, alpha=float(arm["alpha"]))
            if arm.get("log_vectors"):
                li = int(arm.get("log_layer", 22))
                assert 0 <= li < n_layers, f"log_layer {li} out of range"
                tap = PrefillTap(li)
                hookset = HookGroup(hookset, tap)
        try:
            text = _generate(model, tok, device, prompt, hookset, seed)
        except Exception as e:
            print(f"[{arm['id']} trial {i}] ERROR {type(e).__name__}: {e}", flush=True)
            continue
        action, bet_amount, parse_ok, bet_ratio = mw.parse_mw_response(text, balance)
        risky = action == "spin"
        rec = {
            "trial_id": i, "seed": seed, "arm": arm["id"], "phase": phase,
            "mode": mode, "task": "mw", "layers": arm.get("layers"),
            "alpha": arm.get("alpha"), "game_id": int(game_id),
            "bet_type": meta.get("bet_type"), "action": action,
            "amount": bet_amount, "bet_ratio": bet_ratio,
            "risky": bool(risky), "parse_ok": bool(parse_ok),
            "response": text[:300],
            "elapsed_s": round(time.time() - t0, 1),
            "timestamp": datetime.now().isoformat(),
        }
        if tap is not None and tap.hidden is not None:
            rec["vector_log"] = _tap_projection(arm, tap)
        ck.record(rec)
        print(f"  [{arm['id']} {i + 1}/{n}] {action} ${bet_amount} "
              f"({time.time() - t0:.1f}s)", flush=True)
    ck.final_sync()
    _arm_done(arm, ck, _t_start)
    print(f"[{arm['id']}] DONE {len(ck.done_seeds())} trials", flush=True)


def full_layer_list():
    return list(range(N_LAYERS))
