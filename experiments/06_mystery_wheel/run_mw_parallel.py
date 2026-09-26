#!/usr/bin/env python3
"""
Run LLaMA Mystery Wheel experiment in 3 parallel processes on a single A100 80GB.
Each process loads its own LLaMA instance (~16GB) and handles a shard of conditions.
Total VRAM: ~48GB / 80GB available.

Usage:
    python run_mw_parallel.py          # 3 parallel shards
    python run_mw_parallel.py --merge  # merge completed shard JSONs into final
"""
import os, sys, json, subprocess, argparse, time
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("/home/v-seungplee/data/llm-addiction/behavioral/mystery_wheel/llama_v2_role")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All 64 condition combos: 32 prompt_conditions × 2 bet_types
# We shard by (bet_type, condition_index) pairs
# 32 conditions × 2 bet_types = 64 combos, 50 reps each = 3200 games
# Shard into 3: 22 + 21 + 21 combos

def make_shard_script(shard_id: int, combo_start: int, combo_end: int, gpu_id: int = 0):
    """Generate a Python script that runs a shard of the experiment."""
    script = f'''#!/usr/bin/env python3
"""MW shard {shard_id}: combos [{combo_start}, {combo_end})"""
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
sys.path.insert(0, "exploratory_experiments/alternative_paradigms/src")

from mystery_wheel.run_experiment import MysteryWheelExperiment
from common.utils import setup_logger, save_json, set_random_seed
from datetime import datetime
from tqdm import tqdm
import logging

logger = setup_logger(f"mw_shard{shard_id}")
set_random_seed(42)

exp = MysteryWheelExperiment("llama", 0, bet_constraint=30)
exp.load_model()

# Get all combos
prompt_conditions = [name for name, _ in exp.get_prompt_combinations()]
bet_types = ["variable", "fixed"]
repetitions = 50

all_combos = []
for bt in bet_types:
    for pc in prompt_conditions:
        all_combos.append((bt, pc))

COMBO_START = {combo_start}
COMBO_END = {combo_end}
SHARD_ID = {shard_id}

my_combos = all_combos[COMBO_START:COMBO_END]
logger.info(f"Shard {{SHARD_ID}}: {{len(my_combos)}} combos ({{COMBO_START}}-{{COMBO_END}}), {{len(my_combos)*repetitions}} games")

results = []
game_id = COMBO_START * repetitions

for bt, condition in my_combos:
    exp.bet_type = bt
    logger.info(f"  {{bt}}/{{condition}}")
    for rep in tqdm(range(repetitions), desc=f"  {{bt}}/{{condition}}", leave=False):
        game_id += 1
        seed = game_id + 99999
        try:
            result = exp.play_game(condition, game_id, seed)
            results.append(result)
        except Exception as e:
            logger.error(f"    Game {{game_id}} failed: {{e}}")
    logger.info(f"    Done: {{len(results)}} games total")

out_file = Path("{OUT_DIR}") / f"shard_{{SHARD_ID}}.json"
save_json({{
    "shard_id": SHARD_ID,
    "combo_range": [COMBO_START, COMBO_END],
    "n_games": len(results),
    "results": results
}}, out_file)
logger.info(f"Saved {{len(results)}} games to {{out_file}}")
'''
    return script


def launch_shards():
    """Launch 3 parallel shard processes."""
    # 64 combos total, split into 2 shards: 32 + 32
    shards = [(0, 0, 32), (1, 32, 64)]

    procs = []
    for shard_id, start, end in shards:
        script_path = OUT_DIR / f"_shard_{shard_id}.py"
        script_content = make_shard_script(shard_id, start, end, gpu_id=0)
        script_path.write_text(script_content)

        log_path = OUT_DIR / f"shard_{shard_id}.log"
        print(f"Launching shard {shard_id}: combos [{start}, {end}), "
              f"{(end-start)*50} games, log: {log_path}")

        proc = subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            cwd="/home/v-seungplee/llm-addiction"
        )
        procs.append((shard_id, proc, log_path))
        time.sleep(10)  # stagger model loading to avoid VRAM spike

    print(f"\n3 shards launched. PIDs: {[p.pid for _, p, _ in procs]}")
    print(f"Monitor: tail -f {OUT_DIR}/shard_*.log")
    print(f"Check GPU: nvidia-smi")
    print(f"After completion: python run_mw_parallel.py --merge")

    return procs


def merge_shards():
    """Merge completed shard JSONs into final consolidated file."""
    shard_files = sorted(OUT_DIR.glob("shard_*.json"))
    if not shard_files:
        print("No shard files found!")
        return

    all_results = []
    for sf in shard_files:
        data = json.loads(sf.read_text())
        print(f"  {sf.name}: {data['n_games']} games")
        all_results.extend(data["results"])

    # Sort by game_id
    all_results.sort(key=lambda r: r.get("game_id", 0))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final = {
        "experiment": "mystery_wheel",
        "model": "llama",
        "timestamp": timestamp,
        "config": {
            "initial_balance": 100,
            "max_rounds": 100,
            "bet_types": ["variable", "fixed"],
            "bet_constraint": "30",
            "quick_mode": False,
            "total_games": len(all_results),
            "conditions": 32,
            "repetitions": 50,
            "zones": {
                "Red": {"probability": 0.25, "payout": 2.0},
                "Blue": {"probability": 0.08, "payout": 3.0},
                "Gold": {"probability": 0.02, "payout": 8.0},
                "Black": {"probability": 0.65, "payout": 0.0},
            },
            "probability_hidden": True,
            "expected_value": 0.90
        },
        "results": all_results
    }

    out_file = OUT_DIR / f"llama_mysterywheel_c30_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(final, f, indent=2, default=str)

    print(f"\nMerged {len(all_results)} games -> {out_file}")
    print(f"Expected: 3200, Got: {len(all_results)}")

    # BK stats
    bk = sum(1 for r in all_results if r.get("bankruptcy") or r.get("final_outcome") == "bankruptcy")
    print(f"BK: {bk} ({bk/len(all_results)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true", help="Merge completed shards")
    args = parser.parse_args()

    if args.merge:
        merge_shards()
    else:
        launch_shards()
