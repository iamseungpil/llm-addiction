"""All-layer §4 read spine — node driver.

Loops run_spine_layer over (section x task x layer), resume-safe (skips a cell
whose read_profile JSON already exists), and uploads each profile to HF so a
preempted node resumes by re-pulling done profiles. No new statistics: every
cell is computed by run_spine_layer.SECTIONS (the L22-reproducing paper path).

Usage (on node):
  python run_spine_sweep.py --model gemma --sections indicators,condition \
      --tasks slot_machine,investment_choice,mystery_wheel --layers 0-41 \
      --data-root /scratch/spine/data/sae_features_v3 \
      --hf-repo llm-addiction-research/llm-addiction \
      --hf-prefix experiments/spine/profiles
"""
from __future__ import annotations
import argparse, os, json, traceback
from pathlib import Path

import run_spine_layer as rsl

# Isolated all-layer §4.2 handler (NEW section, registered here only — paper
# scripts and run_spine_layer.SECTIONS are untouched). Cross-task is per-(model,
# layer): handled below with a per-layer (not per-task) filename so it never
# collides with the deferred 5-layer 'crosstask' profiles.
import crosstask42_alllayer as ct42
SECTIONS = dict(rsl.SECTIONS)
SECTIONS['crosstask42'] = ct42.section_handler


def parse_layers(spec: str):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def upload(api, repo, prefix, path: Path):
    api.upload_file(path_or_fileobj=str(path),
                    path_in_repo=f"{prefix}/{path.name}",
                    repo_id=repo, repo_type="dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['gemma', 'llama'])
    ap.add_argument('--sections', required=True)
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--layers', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out-dir', default=str(rsl.OUT_DIR))
    ap.add_argument('--hf-repo', default=None)
    ap.add_argument('--hf-prefix', default='experiments/spine/profiles')
    args = ap.parse_args()

    rsl._redirect_data_root(Path(args.data_root))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rsl.OUT_DIR = out_dir  # run_spine_layer's writers use this module-level Path

    api = None
    if args.hf_repo:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi(token=os.environ.get('HF_TOKEN'))
        # Resume across nodes: pull already-computed profiles so skip-if-exists
        # picks up where a paused/preempted run left off (no recompute).
        try:
            snapshot_download(args.hf_repo, repo_type='dataset', local_dir=str(out_dir.parent),
                              token=os.environ.get('HF_TOKEN'),
                              allow_patterns=[f"{args.hf_prefix}/*.json"])
            pulled = out_dir.parent / args.hf_prefix
            if pulled.is_dir() and pulled.resolve() != out_dir.resolve():
                for j in pulled.glob('*.json'):
                    dst = out_dir / j.name
                    if not dst.exists():
                        dst.write_bytes(j.read_bytes())
            print(f'[resume] pulled {len(list(out_dir.glob("*.json")))} prior profiles', flush=True)
        except Exception as e:
            print(f'[resume] none/err: {e}', flush=True)

    sections = args.sections.split(',')
    tasks = args.tasks.split(',')
    layers = parse_layers(args.layers)
    done = skipped = failed = 0
    for section in sections:
        fn = SECTIONS[section]
        # crosstask42 is per-(model, layer) not per-task: run once per layer with
        # a task-free filename. Other sections keep per-(task, layer) cells.
        cell_tasks = [tasks[0]] if section == 'crosstask42' else tasks
        for task in cell_tasks:
            for layer in layers:
                if section == 'crosstask42':
                    name = f'read_profile_{args.model}_crosstask42_L{layer}.json'
                else:
                    name = f'read_profile_{args.model}_{task}_{section}_L{layer}.json'
                path = out_dir / name
                if path.exists():
                    skipped += 1
                    continue
                try:
                    result = fn(args.model, task, layer)
                except Exception as e:
                    failed += 1
                    print(f'[FAIL] {name}: {type(e).__name__}: {e}', flush=True)
                    traceback.print_exc()
                    continue
                payload = {'model': args.model, 'section': section,
                           'layer': layer, 'result': result}
                if section != 'crosstask42':  # crosstask42 is per-(model, layer), task-free
                    payload['task'] = task
                path.write_text(json.dumps(payload, indent=2))
                if api:
                    try:
                        upload(api, args.hf_repo, args.hf_prefix, path)
                    except Exception as e:
                        print(f'[WARN] upload {name}: {e}', flush=True)
                done += 1
                print(f'[OK] {name}', flush=True)
    print(f'SWEEP_DONE done={done} skipped={skipped} failed={failed}', flush=True)


if __name__ == '__main__':
    main()
