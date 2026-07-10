"""Build per-layer rank-128 PCA bases from phase_a_hidden_states.npz (Gemma SM).

Run once before E2 (needs ~16GB RAM + the 6GB npz):
  python -m multilayer_causal.src.pca_basis --dest multilayer_causal/out/pca_bases.npz
Uploads to experiments/multilayer_causal/assets/pca_bases_gemma_sm.npz unless
--no-upload.

NOTE: verify the phase_a array layout against phase_a_metadata.json
(n_rounds=21421, n_layers=42, hidden_dim=3584) on first run; the code asserts
the expected (n, L, D) shape and fails loudly otherwise.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

HF_REPO = "llm-addiction-research/llm-addiction"
SRC = "sae_features_v3/slot_machine/gemma/checkpoint/phase_a_hidden_states.npz"
R_MAX = 128
N_LAYERS, D_MODEL = 42, 3584


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--src-local", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    src = args.src_local
    if src is None:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(HF_REPO, SRC, repo_type="dataset", token=token)
    z = np.load(src, mmap_mode="r")
    key = "hidden_states" if "hidden_states" in z else list(z.keys())[0]
    hs = z[key]
    assert hs.ndim == 3 and hs.shape[1] == N_LAYERS and hs.shape[2] == D_MODEL, \
        f"unexpected phase_a layout: key={key} shape={hs.shape}"
    out = {}
    from sklearn.utils.extmath import randomized_svd
    for l in range(N_LAYERS):
        X = np.asarray(hs[:, l, :], dtype=np.float32)
        X -= X.mean(axis=0, keepdims=True)
        _, _, Vt = randomized_svd(X, n_components=R_MAX, random_state=0)
        out[f"L{l}"] = Vt.T.astype(np.float32)  # (D, R_MAX)
        print(f"L{l} done", flush=True)
    np.savez_compressed(args.dest, **out)
    print(f"bases -> {args.dest}")
    if not args.no_upload and token:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=args.dest,
            path_in_repo="experiments/multilayer_causal/assets/pca_bases_gemma_sm.npz",
            repo_id=HF_REPO, repo_type="dataset",
            commit_message="pca bases gemma sm L0-41 r128")


if __name__ == "__main__":
    main()
