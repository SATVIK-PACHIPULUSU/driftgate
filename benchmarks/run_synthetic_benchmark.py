import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss
from driftgate.core import Gate


def compute_psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref[~np.isnan(ref)], quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref[~np.isnan(ref)], bins=edges)[0] / max(len(ref), 1)
    p_cur = np.histogram(cur[~np.isnan(cur)], bins=edges)[0] / max(len(cur), 1)
    p_ref, p_cur = np.clip(p_ref, 1e-4, 1.0), np.clip(p_cur, 1e-4, 1.0)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def make_dataset(n: int, rng: np.random.Generator, mode: str = "clean") -> tuple[pd.DataFrame, np.ndarray]:
    # x0: Bimodal feature. Active splits exist ONLY below x0 < 2.0. Upper mode (x0 > 3) is a flat plateau.
    is_upper = rng.random(n) < 0.40
    upper_mean = 8.0 if mode == "plateau_shift" else 5.0
    x0 = np.where(is_upper, rng.normal(upper_mean, 0.5, n), rng.normal(0.0, 1.0, n))

    # x1, x2: Correlated interaction pair (marginal N(0,1) preserved under correlation flip)
    rho = -0.8 if mode == "interaction_flip" else 0.8
    cov = [[1.0, rho], [rho, 1.0]]
    x12 = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    x1, x2 = x12[:, 0], x12[:, 1]

    if mode == "cliff_shift":
        # Subtle +0.48σ shift on lower mode: keeps raw PSI < 0.15 (below 0.20 alarm),
        # but pushes ~15% of rows directly across the x0 = 0.5 tree split cliff!
        x0 = np.where(is_upper, x0, x0 + 0.48)

    # 17 pure noise features
    noise = rng.normal(0, 1, size=(n, 17))
    if mode == "noise_shift":
        noise[:, :5] += 2.5

    X = pd.DataFrame(
        np.column_stack([x0, x1, x2, noise]),
        columns=["x0_plateau", "x1_inter", "x2_inter"] + [f"noise_{i}" for i in range(17)],
    )

    # True Data Generating Process: Sharp cliff at x0 > 0.5 + strong x1*x2 interaction
    logit = 2.4 * (x0 > 0.95).astype(float) - 1.2 + 1.8 * (x1 * x2)
    if mode == "cliff_shift":
        # Boundary region around the cliff experiences calibration degradation
        logit = 3.6 * (x0 > 0.25).astype(float) - 1.8 + 1.8 * (x1 * x2)
    elif mode == "concept_flip":
        logit = -logit  # Pure P(Y|X) inversion with identical P(X)

    prob = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, prob)

    if mode == "etl_nan_bug":
        mask = rng.random(n) < 0.18
        X.loc[mask, "x0_plateau"] = np.nan

    return X, y


def run_benchmark():
    rng = np.random.default_rng(123)
    X_train, y_train = make_dataset(15000, rng, "clean")
    X_ref, y_ref = make_dataset(8000, rng, "clean")

    # Properly regularized LightGBM (colsample_bytree=1.0 + min_split_gain=6.0 so trees don't split on pure noise)
    model = lgb.LGBMClassifier(
        n_estimators=60,
        max_depth=4,
        num_leaves=10,
        min_child_samples=200,
        min_split_gain=6.0,
        reg_alpha=10.0,
        colsample_bytree=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    base_brier = brier_score_loss(y_ref, model.predict_proba(X_ref)[:, 1])

    # Fit DriftGate on held-out reference data
    gate = Gate.fit(model, X_ref, n_bootstraps=600, alpha=0.01, seed=42)

    # Gain-weighted PSI baseline
    gains = model.booster_.feature_importance(importance_type="gain")
    gains = gains / np.sum(gains)

    scenarios = [
        ("1a. Clean Window (N=2000)", "clean", 2000),
        ("1b. Clean Small Window (N=500)", "clean", 500),
        ("2a. Benign Noise Shift (+2.5σ)", "noise_shift", 2000),
        ("2b. Saturated Leaf Plateau Shift", "plateau_shift", 2000),
        ("3a. Upstream ETL NaN Bug (18%)", "etl_nan_bug", 2000),
        ("3b. True Decision Cliff Shift", "cliff_shift", 2000),
        ("3c. Correlation Flip (x1*x2)", "interaction_flip", 2000),
        ("4.  Pure Concept Drift (Staleness)", "concept_flip", 2000),
    ]

    print(f"{'Scenario':<34} | {'Δ Brier':<8} | {'Max PSI':<7} | {'Gain-PSI':<8} | {'Gate T':<6} | {'Verdict (Exit Code)'}")
    print("-" * 105)

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        for name, mode, n_rows in scenarios:
            X_cur, y_cur = make_dataset(n_rows, rng, mode)
            brier_delta = brier_score_loss(y_cur, model.predict_proba(X_cur)[:, 1]) - base_brier

            psis = np.array([compute_psi(X_ref[c].to_numpy(), X_cur[c].to_numpy()) for c in X_ref.columns])
            max_psi = float(np.max(psis))
            gain_psi = float(np.sum(psis * gains))

            # Demonstrate max_consecutive_skips=2 triggering on Scenario 4
            if mode == "concept_flip":
                gate.check(X_cur, max_consecutive_skips=2, state_file=state_path)
                rep = gate.check(X_cur, max_consecutive_skips=2, state_file=state_path)
            else:
                rep = gate.check(X_cur)

            print(
                f"{name:<34} | {brier_delta:+8.4f} | {max_psi:7.3f} | {gain_psi:8.3f} | "
                f"{rep.joint_stat_T:6.2f} | {rep.verdict} ({rep.exit_code})"
            )


if __name__ == "__main__":
    run_benchmark()