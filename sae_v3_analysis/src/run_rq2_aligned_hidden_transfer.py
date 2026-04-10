#!/usr/bin/env python3
"""
RQ2 aligned hidden-state transfer and readout decomposition.

Goal
----
Test whether cross-task common structure is better described as a shared
low-rank hidden geometry with task-specific readouts, rather than direct reuse
of sparse features or raw local directions.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


ANALYSIS_ROOT = Path(
    os.environ.get(
        "LLM_ADDICTION_ANALYSIS_ROOT",
        "/home/v-seungplee/llm-addiction/sae_v3_analysis",
    )
)
DATA_ROOT = Path(
    os.environ.get(
        "LLM_ADDICTION_DATA_ROOT",
        "/home/v-seungplee/data/llm-addiction/sae_features_v3",
    )
)
RESULTS_DIR = ANALYSIS_ROOT / "results" / "robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_DIRS = {
    "sm": "slot_machine",
    "ic": "investment_choice",
    "mw": "mystery_wheel",
}

DEFAULT_TASKS = {
    "gemma": ["ic", "sm", "mw"],
    "llama": ["ic", "sm"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "llama"], required=True)
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task list")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--basis-method",
        choices=["readout_pca", "centroid_pca"],
        default="readout_pca",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-tag",
        type=str,
        default="",
        help="Optional tag appended to the output filename for sweep runs.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_hidden_states_dp(model, paradigm, layer):
    path = DATA_ROOT / TASK_DIRS[paradigm] / model / "hidden_states_dp.npz"
    if not path.exists():
        return None

    raw = np.load(path, allow_pickle=True)
    layers = raw["layers"]
    layer_matches = np.where(layers == layer)[0]
    if len(layer_matches) == 0:
        raise ValueError(f"Layer {layer} not found in {path}")
    layer_idx = int(layer_matches[0])

    hidden = raw["hidden_states"][:, layer_idx, :].astype(np.float32)
    meta = {
        "game_ids": raw["game_ids"],
        "game_outcomes": raw["game_outcomes"],
    }
    for field in ["bet_types", "prompt_conditions", "balances", "round_nums"]:
        if field in raw.files:
            meta[field] = raw[field]
    return hidden, meta


def get_bk_labels(meta):
    return (meta["game_outcomes"] == "bankruptcy").astype(np.int32)


def build_task_dataset(model, paradigm, layer):
    loaded = load_hidden_states_dp(model, paradigm, layer)
    if loaded is None:
        return None

    hidden, meta = loaded
    labels = get_bk_labels(meta)
    if labels.sum() < 5 or (1 - labels).sum() < 5:
        return None

    return {
        "task": paradigm,
        "X": hidden,
        "y": labels,
        "meta": meta,
    }


def fit_task_readout(X_train, y_train, pca_dim, seed):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    n_comp = min(pca_dim, X_train_s.shape[0] - 1, X_train_s.shape[1])
    if n_comp < 1:
        raise ValueError("PCA dimension collapsed below 1")
    pca = PCA(n_components=n_comp, random_state=seed)
    X_train_p = pca.fit_transform(X_train_s)

    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(X_train_p, y_train)

    pca_space_readout = clf.coef_[0].astype(np.float64)
    original_space_readout = pca.components_.T @ pca_space_readout
    return {
        "scaler": scaler,
        "pca": pca,
        "clf": clf,
        "readout": original_space_readout.astype(np.float64),
        "pca_space_readout": pca_space_readout,
    }


def centroid_direction(X_train, y_train):
    return X_train[y_train == 1].mean(axis=0) - X_train[y_train == 0].mean(axis=0)


def make_unit(v):
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros_like(v)
    return v / norm


def build_shared_basis(train_payloads, rank, method, seed):
    vectors = []
    for payload in train_payloads.values():
        if method == "readout_pca":
            vectors.append(payload["readout_model"]["readout"])
        elif method == "centroid_pca":
            vectors.append(payload["centroid_direction"])
        else:
            raise ValueError(f"Unknown basis method: {method}")

    matrix = np.stack(vectors, axis=0)
    basis_rank = min(rank, matrix.shape[0], matrix.shape[1])
    if basis_rank < 1:
        raise ValueError("Shared basis rank collapsed below 1")
    if matrix.shape[0] == 1:
        unit = make_unit(matrix[0]).astype(np.float64)
        return unit[:, None], np.array([1.0], dtype=np.float64)

    pca = PCA(n_components=basis_rank, random_state=seed)
    pca.fit(matrix)
    basis = pca.components_.T.astype(np.float64)
    return basis, pca.explained_variance_ratio_.astype(np.float64)


def random_orthogonal_basis(dim, rank, seed):
    rng = np.random.RandomState(seed)
    mat = rng.randn(dim, rank)
    q, _ = np.linalg.qr(mat)
    return q[:, :rank].astype(np.float64)


def project_scores(X, basis, vector):
    basis = basis.astype(np.float64)
    vector = vector.astype(np.float64)
    coeffs = basis.T @ vector
    projected = basis @ coeffs
    scores = X.astype(np.float64) @ projected
    return scores, projected


def fit_basis_readout_auc(X_train, y_train, X_test, y_test, basis, seed):
    Z_train = X_train.astype(np.float64) @ basis.astype(np.float64)
    Z_test = X_test.astype(np.float64) @ basis.astype(np.float64)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(Z_train, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(Z_test)[:, 1]))


def eval_readout_model(readout_model, X_test, y_test):
    X_test_s = readout_model["scaler"].transform(X_test)
    X_test_p = readout_model["pca"].transform(X_test_s)
    return float(roc_auc_score(y_test, readout_model["clf"].predict_proba(X_test_p)[:, 1]))


def auc_from_scores(y_true, scores):
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def decompose_readout(X_test, y_test, readout, basis):
    readout = readout.astype(np.float64)
    basis = basis.astype(np.float64)
    coeffs = basis.T @ readout
    shared = basis @ coeffs
    residual = readout - shared

    norm = np.linalg.norm(readout)
    shared_frac = float(np.linalg.norm(shared) / norm) if norm > 1e-12 else np.nan
    residual_frac = float(np.linalg.norm(residual) / norm) if norm > 1e-12 else np.nan

    return {
        "shared_norm_fraction": shared_frac,
        "residual_norm_fraction": residual_frac,
        "full_auc": auc_from_scores(y_test, X_test @ readout),
        "shared_only_auc": auc_from_scores(y_test, X_test @ shared),
        "residual_only_auc": auc_from_scores(y_test, X_test @ residual),
    }


def run_one_split(model, tasks, layer, rank, pca_dim, basis_method, split_idx, seed):
    split_seed = seed + split_idx
    datasets = {}
    splitters = {}
    indices = {}

    for task in tasks:
        ds = build_task_dataset(model, task, layer)
        if ds is None:
            continue
        datasets[task] = ds
        splitters[task] = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.35,
            random_state=split_seed,
        )
        train_idx, test_idx = next(splitters[task].split(ds["X"], ds["y"]))
        indices[task] = {"train": train_idx, "test": test_idx}

    if len(datasets) < 2:
        raise ValueError("Need at least two tasks with usable hidden-state datasets")

    train_payloads = {}
    for task, ds in datasets.items():
        tr = indices[task]["train"]
        X_train = ds["X"][tr]
        y_train = ds["y"][tr]
        train_payloads[task] = {
            "X_train": X_train,
            "y_train": y_train,
            "readout_model": fit_task_readout(X_train, y_train, pca_dim=pca_dim, seed=split_seed),
            "centroid_direction": centroid_direction(X_train, y_train).astype(np.float64),
        }

    feature_dim = next(iter(train_payloads.values()))["X_train"].shape[1]
    full_shared_basis, full_explained = build_shared_basis(
        train_payloads,
        rank=rank,
        method=basis_method,
        seed=split_seed,
    )

    results = {
        "split_index": split_idx,
        "basis_method": basis_method,
        "rank": int(full_shared_basis.shape[1]),
        "shared_basis_explained_variance": full_explained.tolist(),
        "pairwise_transfer": {},
        "readout_decomposition": {},
    }

    tasks_available = list(datasets.keys())
    for src in tasks_available:
        for tgt in tasks_available:
            if src == tgt:
                continue
            pair_key = f"{src}_to_{tgt}"
            src_payload = train_payloads[src]
            tgt_ds = datasets[tgt]
            tgt_train_idx = indices[tgt]["train"]
            tgt_test_idx = indices[tgt]["test"]
            X_tgt_train = tgt_ds["X"][tgt_train_idx]
            y_tgt_train = tgt_ds["y"][tgt_train_idx]
            X_tgt_test = tgt_ds["X"][tgt_test_idx]
            y_tgt_test = tgt_ds["y"][tgt_test_idx]

            src_readout = src_payload["readout_model"]["readout"]
            raw_scores = X_tgt_test @ src_readout
            basis_tasks = {
                task: payload
                for task, payload in train_payloads.items()
                if task != tgt
            }
            pair_basis, pair_explained = build_shared_basis(
                basis_tasks,
                rank=rank,
                method=basis_method,
                seed=split_seed + 100 * len(pair_key),
            )
            random_basis = random_orthogonal_basis(
                feature_dim,
                pair_basis.shape[1],
                seed=10000 + split_seed + 100 * len(pair_key),
            )

            rotated_keys = list(basis_tasks.keys())
            rotated_values = [basis_tasks[k]["readout_model"]["readout"] for k in rotated_keys]
            rotated_values = rotated_values[1:] + rotated_values[:1]
            shuffled_basis_payloads = {}
            for i, key in enumerate(rotated_keys):
                shuffled_basis_payloads[key] = {
                    "readout_model": {"readout": rotated_values[i]},
                    "centroid_direction": basis_tasks[key]["centroid_direction"],
                }
            shuffled_basis, _ = build_shared_basis(
                shuffled_basis_payloads,
                rank=rank,
                method="readout_pca",
                seed=20000 + split_seed + 100 * len(pair_key),
            )

            aligned_target_auc = fit_basis_readout_auc(
                X_tgt_train,
                y_tgt_train,
                X_tgt_test,
                y_tgt_test,
                pair_basis,
                seed=split_seed,
            )
            random_target_auc = fit_basis_readout_auc(
                X_tgt_train,
                y_tgt_train,
                X_tgt_test,
                y_tgt_test,
                random_basis,
                seed=split_seed,
            )
            shuffled_target_auc = fit_basis_readout_auc(
                X_tgt_train,
                y_tgt_train,
                X_tgt_test,
                y_tgt_test,
                shuffled_basis,
                seed=split_seed,
            )
            target_full_auc = eval_readout_model(
                train_payloads[tgt]["readout_model"],
                X_tgt_test,
                y_tgt_test,
            )
            _, aligned_vec = project_scores(X_tgt_test, pair_basis, src_readout)

            results["pairwise_transfer"][pair_key] = {
                "raw_source_transfer_auc": auc_from_scores(y_tgt_test, raw_scores),
                "target_full_readout_auc": target_full_auc,
                "aligned_basis_target_readout_auc": aligned_target_auc,
                "random_basis_target_readout_auc": random_target_auc,
                "label_shuffled_basis_target_readout_auc": shuffled_target_auc,
                "aligned_minus_random": aligned_target_auc - random_target_auc,
                "aligned_minus_shuffled": aligned_target_auc - shuffled_target_auc,
                "aligned_minus_full": aligned_target_auc - target_full_auc,
                "aligned_vector_norm": float(np.linalg.norm(aligned_vec)),
                "source_readout_norm": float(np.linalg.norm(src_readout)),
                "basis_rank": int(pair_basis.shape[1]),
                "basis_explained_variance": pair_explained.tolist(),
            }

    for task in tasks_available:
        task_ds = datasets[task]
        task_test_idx = indices[task]["test"]
        X_task_test = task_ds["X"][task_test_idx].astype(np.float64)
        y_task_test = task_ds["y"][task_test_idx]
        task_readout = train_payloads[task]["readout_model"]["readout"]
        results["readout_decomposition"][task] = decompose_readout(
            X_task_test,
            y_task_test,
            task_readout,
            full_shared_basis,
        )

    return results


def summarize_splits(split_results):
    summary = {
        "pairwise_transfer": {},
        "readout_decomposition": {},
    }

    pair_keys = sorted(split_results[0]["pairwise_transfer"].keys())
    for key in pair_keys:
        metrics = split_results[0]["pairwise_transfer"][key].keys()
        summary["pairwise_transfer"][key] = {}
        for metric in metrics:
            vals = np.array([s["pairwise_transfer"][key][metric] for s in split_results], dtype=float)
            summary["pairwise_transfer"][key][f"{metric}_mean"] = float(np.nanmean(vals))
            summary["pairwise_transfer"][key][f"{metric}_std"] = float(np.nanstd(vals))

    task_keys = sorted(split_results[0]["readout_decomposition"].keys())
    for key in task_keys:
        metrics = split_results[0]["readout_decomposition"][key].keys()
        summary["readout_decomposition"][key] = {}
        for metric in metrics:
            vals = np.array([s["readout_decomposition"][key][metric] for s in split_results], dtype=float)
            summary["readout_decomposition"][key][f"{metric}_mean"] = float(np.nanmean(vals))
            summary["readout_decomposition"][key][f"{metric}_std"] = float(np.nanstd(vals))

    return summary


def main():
    args = parse_args()
    tasks = DEFAULT_TASKS[args.model] if args.tasks is None else args.tasks.split(",")
    n_splits = 2 if args.smoke else args.n_splits

    split_results = []
    for split_idx in range(n_splits):
        split_results.append(
            run_one_split(
                model=args.model,
                tasks=tasks,
                layer=args.layer,
                rank=args.rank,
                pca_dim=args.pca_dim,
                basis_method=args.basis_method,
                split_idx=split_idx,
                seed=args.seed,
            )
        )

    payload = {
        "config": {
            "model": args.model,
            "tasks": tasks,
            "layer": args.layer,
            "rank": args.rank,
            "pca_dim": args.pca_dim,
            "n_splits": n_splits,
            "basis_method": args.basis_method,
            "smoke": bool(args.smoke),
        },
        "split_results": split_results,
        "summary": summarize_splits(split_results),
    }

    suffix_parts = [
        args.model,
        args.basis_method,
        f"L{args.layer}",
        f"r{args.rank}",
    ]
    if args.out_tag:
        suffix_parts.append(args.out_tag)
    suffix = "_".join(suffix_parts)
    out_name = (
        f"rq2_aligned_hidden_transfer_{suffix}_smoke.json"
        if args.smoke
        else f"rq2_aligned_hidden_transfer_{suffix}.json"
    )
    out_path = RESULTS_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload["summary"], indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
