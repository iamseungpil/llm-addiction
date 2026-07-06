"""Per-arm JSONL checkpointing with HF latest-state sync (preemption-safe).

HF layout (dataset llm-addiction-research/llm-addiction):
  experiments/multilayer_causal/checkpoints/{phase}/{arm}.jsonl   <- overwritten
  experiments/multilayer_causal/checkpoints/{phase}/{arm}_vectors.npz
The same path is overwritten each sync: the dataset always shows the LATEST
resumable state (git history keeps the trail). Jobs download their checkpoint
at start, so a preempted job resumes by plain resubmission.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HF_REPO = "llm-addiction-research/llm-addiction"
# HF write prefix. The xtask wave overrides this to experiments/xtask/checkpoints
# via MLC_HF_BASE (additive — with the var unset every existing arm is
# byte-identical on HF).
HF_BASE = os.environ.get("MLC_HF_BASE", "experiments/multilayer_causal/checkpoints")


class ArmCheckpoint:
    def __init__(self, phase, arm_id, out_dir, sync_every=10, hf_enabled=None):
        # MLC_SYNC_EVERY override (additive; unset -> byte-identical default).
        # HF caps repo commits at 256/h: a 69-arm x n200 wave at sync_every=10
        # needs ~1.4k commits and throttles itself; a larger interval trades at
        # most that many trials on preemption for staying under the cap.
        sync_every = int(os.environ.get("MLC_SYNC_EVERY", sync_every))
        self.phase, self.arm_id = phase, arm_id
        self.path = Path(out_dir) / f"{arm_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hf_path = f"{HF_BASE}/{phase}/{arm_id}.jsonl"
        self.token = os.environ.get("HF_TOKEN")
        self.hf_enabled = bool(self.token) if hf_enabled is None else hf_enabled
        self.sync_every = sync_every
        self._since = 0
        if not self.path.exists() and self.hf_enabled:
            self._download()

    def _download(self):
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(HF_REPO, self.hf_path, repo_type="dataset",
                                token=self.token)
            self.path.write_bytes(Path(p).read_bytes())
            print(f"[ckpt] resumed {self.arm_id} from HF "
                  f"({len(self.done_seeds())} trials)", flush=True)
        except Exception:
            pass  # nothing to resume

    def done_seeds(self):
        seeds = set()
        if self.path.exists():
            for line in open(self.path):
                try:
                    seeds.add(json.loads(line)["seed"])
                except Exception:
                    pass
        return seeds

    def record(self, result):
        with open(self.path, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._since += 1
        if self.hf_enabled and self._since % self.sync_every == 0:
            self._upload()

    def final_sync(self):
        if self.hf_enabled:
            self._upload()

    def record_summary(self, meta):
        """Write per-arm run-metadata ({n, t_start, t_end, replay_hash, ...})
        to {arm_id}_summary.json (additive — sits beside the jsonl, never
        rewrites it) and mirror it to HF. Provenance guard against 10-second
        smoke JSONs being mistaken for real runs (audit: provenance drift)."""
        sp = self.path.with_name(f"{self.arm_id}_summary.json")
        sp.write_text(json.dumps(meta, indent=1))
        if self.hf_enabled:
            try:
                from huggingface_hub import HfApi
                HfApi(token=self.token).upload_file(
                    path_or_fileobj=str(sp),
                    path_in_repo=f"{HF_BASE}/{self.phase}/{self.arm_id}_summary.json",
                    repo_id=HF_REPO, repo_type="dataset",
                    commit_message=f"summary {self.phase}/{self.arm_id}")
            except Exception as e:
                print(f"[ckpt] summary sync failed: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)

    def _upload(self):
        try:
            from huggingface_hub import HfApi
            HfApi(token=self.token).upload_file(
                path_or_fileobj=str(self.path), path_in_repo=self.hf_path,
                repo_id=HF_REPO, repo_type="dataset",
                commit_message=f"ckpt {self.phase}/{self.arm_id}")
        except Exception as e:  # never kill the run over a sync hiccup
            print(f"[ckpt] HF sync failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)


class VectorStore:
    """Per-arm (n, L, D) fp16 stacks of −G / +G last-token hidden vectors."""

    def __init__(self, path, n_layers, d_model):
        self.path = Path(path)
        self.n_layers, self.d_model = n_layers, d_model
        if self.path.exists():
            z = np.load(self.path)
            self.seeds = [int(s) for s in z["seeds"]]
            self._minus = [m for m in z["minus"]]
            self._plus = [p for p in z["plus"]]
        else:
            self.seeds, self._minus, self._plus = [], [], []

    @property
    def minus(self):
        if not self._minus:
            return np.zeros((0, self.n_layers, self.d_model), dtype=np.float16)
        return np.stack(self._minus)

    @property
    def plus(self):
        if not self._plus:
            return np.zeros((0, self.n_layers, self.d_model), dtype=np.float16)
        return np.stack(self._plus)

    def append(self, seed, minus, plus):
        assert minus.shape == (self.n_layers, self.d_model)
        assert plus.shape == (self.n_layers, self.d_model)
        self.seeds.append(int(seed))
        self._minus.append(minus.astype(np.float16))
        self._plus.append(plus.astype(np.float16))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, seeds=np.array(self.seeds),
                            minus=self.minus.astype(np.float16),
                            plus=self.plus.astype(np.float16))
