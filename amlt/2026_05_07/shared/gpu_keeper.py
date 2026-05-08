#!/usr/bin/env python3
import os
import signal
import sys
import time
from datetime import datetime, timezone

import torch

INTERVAL_S = int(os.environ.get("GPU_KEEPER_INTERVAL_S", "60"))
_running = True


def _stop(signum, frame):
    global _running
    _running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def main() -> None:
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus == 0:
        print("[gpu_keeper] no cuda; exiting", flush=True)
        return
    tensors = []
    for i in range(n_gpus):
        torch.cuda.set_device(i)
        tensors.append(torch.randn(512, 512, device=f"cuda:{i}"))
    while _running:
        for i, t in enumerate(tensors):
            torch.cuda.set_device(i)
            _ = (t @ t).sum().item()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[gpu_keeper] heartbeat {ts} n_gpus={n_gpus}", flush=True)
        for _ in range(INTERVAL_S):
            if not _running:
                break
            time.sleep(1)
    print("[gpu_keeper] graceful exit", flush=True)


if __name__ == "__main__":
    sys.exit(main())
