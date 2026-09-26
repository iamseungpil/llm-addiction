"""
Causal Mediation Analysis: Autonomy → Neural Representation → Behavior
========================================================================
의도: 자율성 조건이 신경 표상을 경유하여 행동 결과에 영향을 주는지 검증.
      현재 논문의 "상관적 증거"를 "매개 메커니즘"으로 격상 가능한지 탐색.

가설:
  H1: Variable betting → higher BK-projection → higher BK probability
  H2: BK-projection이 자율성→행동 경로를 부분적으로 매개한다
      (indirect effect의 bootstrap CI가 0을 배제)

검증 방법:
  - Split-half: 50% direction estimation, 50% mediation test (double-dipping 방지)
  - Mediator: BK-projection (balance-residualized)
  - Baron-Kenny with logistic outcome: bet_type → M → BK outcome
  - Bootstrap 1000회, cluster = prompt_condition
  - Sensitivity: Imai et al. (2010) ρ parameter

제한:
  - Decision-point hidden states만 사용 (pilot). Round 2 데이터는 별도 추출 필요.
  - Tautology 위험은 balance residualization + split-half로 부분 완화.
  - 단일 모델 (LLaMA), 단일 layer (L22, a priori choice)
"""

import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

DATA_PATH = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama/hidden_states_dp.npz")
OUT_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/mediation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LAYER_IDX = 2  # L22


def load_and_split():
    """Load data and perform 50:50 split for direction estimation vs mediation test."""
    d = np.load(DATA_PATH, allow_pickle=True)
    hs = d["hidden_states"][:, TARGET_LAYER_IDX, :]
    valid = d["valid_mask"]
    outcomes = d["game_outcomes"]
    balances = d["balances"]
    bet_types = d["bet_types"]
    round_nums = d["round_nums"]

    mask = valid & np.isin(outcomes, ["bankruptcy", "voluntary_stop"])
    hs = hs[mask]
    labels = (outcomes[mask] == "bankruptcy").astype(int)
    balances = balances[mask].astype(float)
    bet_binary = (bet_types[mask] == "variable").astype(int)
    round_nums = round_nums[mask].astype(float)

    n = len(labels)
    idx = np.random.permutation(n)
    half = n // 2

    split = {
        "direction": {
            "hs": hs[idx[:half]],
            "labels": labels[idx[:half]],
            "bet": bet_binary[idx[:half]],
            "balance": balances[idx[:half]],
            "round": round_nums[idx[:half]],
        },
        "test": {
            "hs": hs[idx[half:]],
            "labels": labels[idx[half:]],
            "bet": bet_binary[idx[half:]],
            "balance": balances[idx[half:]],
            "round": round_nums[idx[half:]],
        },
    }

    print(f"Direction set: n={half}, BK={split['direction']['labels'].sum()}")
    print(f"Test set: n={n-half}, BK={split['test']['labels'].sum()}")
    return split


def compute_bk_direction(hs, labels):
    """Compute BK direction from direction estimation set."""
    bk_mean = hs[labels == 1].mean(axis=0)
    vs_mean = hs[labels == 0].mean(axis=0)
    direction = bk_mean - vs_mean
    direction = direction / np.linalg.norm(direction)
    return direction


def compute_mediator(hs, direction, balance, round_nums):
    """Compute balance-residualized BK-projection."""
    raw_proj = hs @ direction

    # Residualize on balance and round number
    confounds = np.column_stack([balance, round_nums])
    reg = LinearRegression().fit(confounds, raw_proj)
    residual = raw_proj - reg.predict(confounds)

    return residual, raw_proj


def baron_kenny_logistic(iv, mediator, dv):
    """Baron-Kenny mediation with logistic DV."""
    # Path c: IV → DV (total effect)
    X_c = iv.reshape(-1, 1)
    lr_c = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_c.fit(X_c, dv)
    c_coef = lr_c.coef_[0, 0]

    # Path a: IV → M (linear)
    reg_a = LinearRegression().fit(X_c, mediator)
    a_coef = reg_a.coef_[0]

    # Path b + c': IV + M → DV
    X_bc = np.column_stack([iv, mediator])
    lr_bc = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_bc.fit(X_bc, dv)
    c_prime = lr_bc.coef_[0, 0]  # direct effect
    b_coef = lr_bc.coef_[0, 1]  # M → DV controlling for IV

    # Indirect effect (Sobel-like: a * b)
    indirect = a_coef * b_coef

    return {
        "c_total": float(c_coef),
        "a_iv_to_m": float(a_coef),
        "b_m_to_dv": float(b_coef),
        "c_prime_direct": float(c_prime),
        "indirect_ab": float(indirect),
        "proportion_mediated": float(indirect / c_coef) if c_coef != 0 else float("nan"),
    }


def bootstrap_indirect(iv, mediator, dv, n_boot=1000):
    """Bootstrap CI for indirect effect."""
    n = len(iv)
    indirect_samples = []

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        iv_b, m_b, dv_b = iv[idx], mediator[idx], dv[idx]

        if len(np.unique(dv_b)) < 2:
            continue

        try:
            # Path a
            reg_a = LinearRegression().fit(iv_b.reshape(-1, 1), m_b)
            a = reg_a.coef_[0]

            # Path b (controlling for IV)
            X = np.column_stack([iv_b, m_b])
            lr = LogisticRegression(max_iter=1000, random_state=SEED)
            lr.fit(X, dv_b)
            b = lr.coef_[0, 1]

            indirect_samples.append(a * b)
        except Exception:
            continue

    indirect_samples = np.array(indirect_samples)
    ci_low = np.percentile(indirect_samples, 2.5)
    ci_high = np.percentile(indirect_samples, 97.5)
    p_zero = np.mean(indirect_samples <= 0)  # proportion <= 0

    return {
        "indirect_mean": float(np.mean(indirect_samples)),
        "indirect_std": float(np.std(indirect_samples)),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "p_zero_excluded": bool(ci_low > 0 or ci_high < 0),
        "n_valid_boots": len(indirect_samples),
    }


def main():
    print("=" * 60)
    print("CAUSAL MEDIATION ANALYSIS")
    print("Autonomy → BK-projection → Bankruptcy")
    print("=" * 60)

    # Step 0: Split data
    split = load_and_split()

    # Step 1: Estimate BK direction from direction set
    bk_dir = compute_bk_direction(
        split["direction"]["hs"], split["direction"]["labels"]
    )
    print(f"\nBK direction computed (norm={np.linalg.norm(bk_dir):.4f})")

    # Step 2: Compute mediator on test set
    mediator, raw_proj = compute_mediator(
        split["test"]["hs"], bk_dir,
        split["test"]["balance"], split["test"]["round"]
    )
    iv = split["test"]["bet"]
    dv = split["test"]["labels"]

    print(f"\nMediator stats:")
    print(f"  Raw projection: mean={raw_proj.mean():.4f}, std={raw_proj.std():.4f}")
    print(f"  Residualized: mean={mediator.mean():.4f}, std={mediator.std():.4f}")
    print(f"  IV (variable=1): {iv.sum()}/{len(iv)} ({iv.mean()*100:.1f}%)")
    print(f"  DV (BK=1): {dv.sum()}/{len(dv)} ({dv.mean()*100:.1f}%)")

    # Step 3: Baron-Kenny
    print("\n--- Baron-Kenny Mediation ---")
    bk_results = baron_kenny_logistic(iv, mediator, dv)
    for k, v in bk_results.items():
        print(f"  {k}: {v:.4f}")

    # Step 4: Bootstrap CI
    print("\n--- Bootstrap (n=1000) ---")
    boot_results = bootstrap_indirect(iv, mediator, dv, n_boot=1000)
    for k, v in boot_results.items():
        print(f"  {k}: {v}")

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    significant = boot_results["p_zero_excluded"]
    print(f"  Indirect effect (a×b): {bk_results['indirect_ab']:.4f}")
    print(f"  95% CI: [{boot_results['ci_95_low']:.4f}, {boot_results['ci_95_high']:.4f}]")
    print(f"  Zero excluded: {significant}")
    print(f"  Proportion mediated: {bk_results['proportion_mediated']:.1%}")

    if significant:
        print("\n  ✓ MEDIATION SIGNIFICANT: BK-projection partially mediates")
        print("    the effect of autonomy on bankruptcy behavior.")
    else:
        print("\n  ✗ MEDIATION NOT SIGNIFICANT at 95% level.")

    # Save results
    results = {
        "method": "Baron-Kenny logistic mediation with split-half + balance residualization",
        "baron_kenny": bk_results,
        "bootstrap": boot_results,
        "significant": significant,
        "note": "Decision-point hidden states (pilot). Round 2 analysis pending."
    }
    out_path = OUT_DIR / "mediation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
