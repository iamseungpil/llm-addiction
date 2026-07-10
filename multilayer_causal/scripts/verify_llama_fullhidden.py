#!/usr/bin/env python3
"""Verify the re-extracted LLaMA SM / MW FULL 32-layer phase_a dumps.

After the sec4_w8extract node job uploads
    sae_features_v3/{slot_machine,mystery_wheel}/llama/checkpoints/
        phase_a_hidden_states.npz  (+ phase_a_metadata.json)
this script proves the re-extraction is correct and reproduces the known-good
5-layer game-level dp dump (hidden_states_dp.npz, layers [8,12,22,25,30]):

  (a) new npz hidden_states has shape (N, 32, 4096);
  (b) the npz carries a `layers` provenance array == 0..31 (so
      indicator_axes._load_llama_task_arrays can resolve layer-number -> row);
  (c) for every game-level dp-dump row, keyed by (game_id, round_num), the
      new round-level file has a matching row whose hidden states on the five
      shared layers [8,12,22,25,30] agree within floating-point tolerance.

The new file is ~32-44 GB, so we DO NOT download it: we range-read only the
~3200 matching (row, layer) vectors from the remote UNCOMPRESSED npz (np.savez
writes ZIP_STORED members, so each array is a contiguous .npy blob we can slice
by byte offset). The dp dump is only ~260 MB (game-level) and is pulled whole.

Usage:
    HF_TOKEN=... python multilayer_causal/scripts/verify_llama_fullhidden.py
    # or a single task / custom tolerances:
    python .../verify_llama_fullhidden.py --tasks slot_machine \
        --cos-min 0.999 --rel-l2-max 0.05
"""
from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import zipfile

import numpy as np
from numpy.lib import format as npf

HF_REPO = "llm-addiction-research/llm-addiction"
FULL = {  # the NEW full-layer files this job produced (checkpoints/, plural)
    "slot_machine": "sae_features_v3/slot_machine/llama/checkpoints/"
                    "phase_a_hidden_states.npz",
    "mystery_wheel": "sae_features_v3/mystery_wheel/llama/checkpoints/"
                     "phase_a_hidden_states.npz",
}
DP = {  # the EXISTING known-good 5-layer game-level dp dump
    "slot_machine": "sae_features_v3/slot_machine/llama/hidden_states_dp.npz",
    "mystery_wheel": "sae_features_v3/mystery_wheel/llama/hidden_states_dp.npz",
}
N_LAYERS_FULL, D_MODEL = 32, 4096


class RemoteNpz:
    """Range-reading view over a remote UNCOMPRESSED .npz on the HF hub.

    Small arrays are read whole; one big array can be sliced element-wise by
    byte offset without downloading the multi-GB file.
    """

    def __init__(self, repo_file, token):
        from huggingface_hub import HfFileSystem
        self.fs = HfFileSystem(token=token)
        self.path = f"datasets/{HF_REPO}/{repo_file}"
        self.fh = self.fs.open(self.path, "rb")
        self.zf = zipfile.ZipFile(self.fh)
        self.names = {n[:-4] if n.endswith(".npy") else n: n
                      for n in self.zf.namelist()}

    def keys(self):
        return list(self.names)

    def small(self, key):
        """Read an entire member array (use only for small arrays)."""
        with self.zf.open(self.names[key]) as m:
            return npf.read_array(m, allow_pickle=False)

    def _member_data_start(self, key):
        """Byte offset where the STORED member's raw bytes begin."""
        zi = self.zf.getinfo(self.names[key])
        if zi.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"member {key} is compressed "
                             f"(type {zi.compress_type}); range-read needs "
                             "np.savez (uncompressed)")
        self.fh.seek(zi.header_offset)
        hdr = self.fh.read(30)
        if hdr[:4] != b"PK\x03\x04":
            raise ValueError("bad local file header magic")
        fn_len, extra_len = struct.unpack("<HH", hdr[26:30])
        return zi.header_offset + 30 + fn_len + extra_len

    def big_header(self, key):
        """(data_offset, shape, dtype, fortran) for a big member — data_offset
        is absolute byte offset of the .npy payload in the underlying file."""
        start = self._member_data_start(key)
        self.fh.seek(start)
        # read enough for magic + header (npy headers are <= a few hundred B)
        buf = io.BytesIO(self.fh.read(256))
        ver = npf.read_magic(buf)
        shape, fortran, dtype = npf._read_array_header(buf, ver)
        payload = start + buf.tell()
        return payload, tuple(shape), dtype, fortran

    def read_row_layer(self, payload, shape, dtype, i, l):
        """Read hidden_states[i, l, :] (C-order (N, nL, d) float array)."""
        _, nL, d = shape
        itemsize = np.dtype(dtype).itemsize
        off = payload + (i * nL + l) * d * itemsize
        self.fh.seek(off)
        raw = self.fh.read(d * itemsize)
        return np.frombuffer(raw, dtype=dtype).astype(np.float64)


def verify_task(task, token, cos_min, rel_l2_max, max_report):
    print(f"\n{'='*70}\n[{task}] verifying full-layer re-extraction\n{'='*70}",
          flush=True)
    from huggingface_hub import hf_hub_download

    # --- dp dump (small, game-level) pulled whole -------------------------
    dp_local = hf_hub_download(HF_REPO, DP[task], repo_type="dataset",
                               token=token)
    dp = np.load(dp_local, allow_pickle=True)
    dp_layers = list(np.asarray(dp["layers"]))
    dp_h = np.asarray(dp["hidden_states"])            # (Ndp, 5, 4096)
    dp_g = np.asarray(dp["game_ids"]).astype(np.int64)
    dp_r = np.asarray(dp["round_nums"]).astype(np.int64)
    ndp = dp_h.shape[0]
    print(f"  dp dump: {dp_h.shape}, layers={dp_layers}, N={ndp}", flush=True)

    # --- new full file (huge) opened for range reads ----------------------
    rem = RemoteNpz(FULL[task], token)
    assert "hidden_states" in rem.keys(), \
        f"[{task}] new npz has no hidden_states (keys={rem.keys()})"

    # (a) shape (N, 32, 4096)
    payload, shape, dtype, fortran = rem.big_header("hidden_states")
    assert not fortran, f"[{task}] hidden_states is Fortran-ordered"
    assert len(shape) == 3 and shape[1] == N_LAYERS_FULL \
        and shape[2] == D_MODEL, f"[{task}] bad shape {shape}"
    N = shape[0]
    print(f"  new file: hidden_states {shape} {dtype}  -- SHAPE OK", flush=True)

    # (b) layers == 0..31
    assert "layers" in rem.keys(), \
        f"[{task}] new npz LACKS the `layers` provenance array (keys={rem.keys()})"
    full_layers = list(np.asarray(rem.small("layers")).astype(int))
    assert full_layers == list(range(N_LAYERS_FULL)), \
        f"[{task}] layers != 0..31: {full_layers}"
    print(f"  new file: layers == 0..31  -- LAYERS OK", flush=True)

    # (c) key-align dp rows against new rows and compare shared layers
    assert "game_ids" in rem.keys() and "round_nums" in rem.keys(), \
        (f"[{task}] new npz lacks game_ids/round_nums provenance needed to "
         f"key-align the game-level dp dump (keys={rem.keys()})")
    new_g = np.asarray(rem.small("game_ids")).astype(np.int64)
    new_r = np.asarray(rem.small("round_nums")).astype(np.int64)
    assert new_g.shape[0] == N and new_r.shape[0] == N, \
        f"[{task}] provenance length != N ({new_g.shape[0]} vs {N})"
    index = {}
    for idx in range(N):
        index.setdefault((int(new_g[idx]), int(new_r[idx])), idx)
    print(f"  new file: N={N} round-level rows, "
          f"{len(index)} unique (game_id,round_num) keys", flush=True)

    shared = [L for L in dp_layers if L in full_layers]
    assert shared, f"[{task}] no shared layers between dp {dp_layers} and full"
    dp_row_of = {L: dp_layers.index(L) for L in shared}
    full_row_of = {L: full_layers.index(L) for L in shared}

    missing_keys = 0
    per_layer = {L: {"cos": [], "rel": []} for L in shared}
    worst = []  # (rel_l2, task, layer, game, round)
    for k in range(ndp):
        key = (int(dp_g[k]), int(dp_r[k]))
        j = index.get(key)
        if j is None:
            missing_keys += 1
            continue
        for L in shared:
            a = rem.read_row_layer(payload, shape, dtype, j, full_row_of[L])
            b = dp_h[k, dp_row_of[L], :].astype(np.float64)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            cos = float(a @ b / denom)
            rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
            per_layer[L]["cos"].append(cos)
            per_layer[L]["rel"].append(rel)
            worst.append((rel, L, key[0], key[1]))

    ok = True
    if missing_keys:
        ok = False
        print(f"  !! MISMATCH: {missing_keys}/{ndp} dp (game_id,round_num) keys "
              f"absent from the new file -- alignment/row-set differs", flush=True)
    for L in shared:
        cos = np.asarray(per_layer[L]["cos"])
        rel = np.asarray(per_layer[L]["rel"])
        n_bad_cos = int((cos < cos_min).sum())
        n_bad_rel = int((rel > rel_l2_max).sum())
        status = "OK" if (n_bad_cos == 0 and n_bad_rel == 0) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  L{L:<2} n={len(cos)} cos[min={cos.min():.6f} "
              f"med={np.median(cos):.6f}] relL2[med={np.median(rel):.2e} "
              f"max={rel.max():.2e}] bad_cos={n_bad_cos} bad_rel={n_bad_rel} "
              f"-- {status}", flush=True)

    if not ok:
        worst.sort(reverse=True)
        print(f"  !! [{task}] WORST {min(max_report, len(worst))} rows by relL2:",
              flush=True)
        for rel, L, g, r in worst[:max_report]:
            print(f"       L{L} game={g} round={r} relL2={rel:.3e}", flush=True)
        print(f"  !! [{task}] VERIFY FAILED", flush=True)
    else:
        print(f"  [{task}] VERIFY PASSED "
              f"({ndp}/{ndp} dp rows reproduced on layers {shared})", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Verify llama full-layer phase_a")
    ap.add_argument("--tasks", nargs="+",
                    default=["slot_machine", "mystery_wheel"])
    ap.add_argument("--cos-min", type=float, default=0.999)
    ap.add_argument("--rel-l2-max", type=float, default=0.05)
    ap.add_argument("--max-report", type=int, default=20)
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None

    all_ok = True
    for task in args.tasks:
        assert task in FULL, f"unknown task {task}"
        try:
            all_ok &= verify_task(task, token, args.cos_min, args.rel_l2_max,
                                  args.max_report)
        except AssertionError as e:
            all_ok = False
            print(f"  !! [{task}] ASSERTION: {e}", flush=True)
    print(f"\n{'='*70}\nVERIFY {'PASSED' if all_ok else 'FAILED'} "
          f"for tasks {args.tasks}\n{'='*70}", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
