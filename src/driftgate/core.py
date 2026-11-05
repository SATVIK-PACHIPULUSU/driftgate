from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


class ExitCode(IntEnum):
    SKIP_RETRAIN = 0
    TRIGGER_RETRAIN = 10
    DATA_OR_SCHEMA_ERROR = 20


@dataclass
class SchemaContract:
    features: list[str]
    kinds: dict[str, str]  # 'numeric', 'bool', 'categorical'


@dataclass
class GateReport:
    verdict: str
    exit_code: int
    joint_stat_T: float
    tau_T_p99: float
    mean_w1: float
    max_w1: float
    margin_w1: float
    consecutive_skips: int
    top_drivers: list[dict]
    errors: list[str]

    def summary(self) -> str:
        lines = [
            f"=== DriftGate Verdict: {self.verdict} (Exit Code: {self.exit_code}) ===",
            f"Joint Pivotal Stat (T) : {self.joint_stat_T:.3f} (Threshold p99: {self.tau_T_p99:.3f} -> {self.joint_stat_T / max(self.tau_T_p99, 1e-9):.2f}x)",
            f"Component Drift (W1)   : Mean={self.mean_w1:.4f} | Max={self.max_w1:.4f} | Margin={self.margin_w1:.4f}",
            f"Consecutive Skips      : {self.consecutive_skips}",
        ]
        if self.errors:
            lines.append(f"Pipeline/Data Alerts   : {self.errors}")
        if self.top_drivers:
            lines.append("\nTop Root-Cause Drivers (SHAP Margin Units):")
            for d in self.top_drivers[:3]:
                lines.append(
                    f"  - {d['feature']:<18} | W1: {d['w1_scaled']:.4f} | "
                    f"Rank: #{d['rank_ref']} -> #{d['rank_cur']} | "
                    f"Null Δ: {d['null_delta']:+.1%}"
                )
        return "\n".join(lines)


class _BackendAdapter:
    """Explicit model adapter preventing silent multiclass or API mismatches."""

    @staticmethod
    def compute_shap(model, X: pd.DataFrame) -> np.ndarray:
        mod_type = type(model).__module__
        n_rows, n_feats = X.shape

        # 1. LightGBM (Booster or Sklearn Wrapper)
        if "lightgbm" in mod_type:
            if hasattr(model, "n_classes_") and getattr(model, "n_classes_", 2) > 2:
                raise ValueError("Multiclass LightGBM is unsupported in v0.1. Use binary or regression.")
            contribs = model.predict(X, pred_contrib=True)
            if hasattr(contribs, "toarray"):
                contribs = contribs.toarray()
            contribs = np.asarray(contribs)
            if contribs.ndim != 2 or contribs.shape[1] != n_feats + 1:
                raise ValueError(
                    f"Expected shape ({n_rows}, {n_feats + 1}), got {contribs.shape}. "
                    "Multiclass models are unsupported in v0.1."
                )
            return contribs[:, :-1]

        # 2. XGBoost (Booster or Sklearn Wrapper)
        if "xgboost" in mod_type:
            import xgboost as xgb
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            dm = xgb.DMatrix(X, enable_categorical=True)
            contribs = booster.predict(dm, pred_contribs=True)
            if contribs.ndim != 2 or contribs.shape[1] != n_feats + 1:
                raise ValueError(f"Unsupported XGBoost output shape {contribs.shape}; v0.1 supports binary/regression.")
            return contribs[:, :-1]

        raise TypeError(f"Unsupported model backend: {type(model)}. Supported: LightGBM, XGBoost (binary/regression).")


class Gate:
    def __init__(self, model=None):
        self.model = model
        self.contract: SchemaContract | None = None
        self.ref_shap: np.ndarray | None = None
        self.ref_margin: np.ndarray | None = None
        self.ref_null_rates: dict[str, float] = {}
        self.margin_std: float = 1.0
        self.max_null_delta_tau: float = 0.05

        # Log-log power-law coefficients for sample-size aware null scaling: log(stat) = intercept + slope * log(n)
        self.power_law_medians: dict[str, tuple[float, float]] = {}
        self.tau_T_p99: float = 1.0

    @staticmethod
    def _col_kind(dtype) -> str:
        if pd.api.types.is_bool_dtype(dtype):
            return "bool"
        if pd.api.types.is_numeric_dtype(dtype):
            return "numeric"
        return "categorical"

    @staticmethod
    def _vectorized_equal_w1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Exact, O(N log N) C-speed 1D Wasserstein across all columns for equal sample sizes."""
        return np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).mean(axis=0)

    def _predict_null_medians(self, n: int) -> tuple[float, float, float]:
        log_n = np.log(max(n, 50))
        med_mean = np.exp(self.power_law_medians["mean"][0] + self.power_law_medians["mean"][1] * log_n)
        med_max = np.exp(self.power_law_medians["max"][0] + self.power_law_medians["max"][1] * log_n)
        med_margin = np.exp(self.power_law_medians["margin"][0] + self.power_law_medians["margin"][1] * log_n)
        return float(med_mean), float(med_max), float(med_margin)

    @classmethod
    def fit(
        cls,
        model,
        reference: pd.DataFrame,
        n_bootstraps: int = 1000,
        alpha: float = 0.01,
        null_delta_floor: float = 0.05,
        seed: int = 42,
    ) -> "Gate":
        if len(reference) < 1000:
            raise ValueError("Reference holdout must contain at least 1,000 rows for disjoint null calibration.")

        gate = cls(model=model)
        gate.contract = SchemaContract(
            features=list(reference.columns),
            kinds={col: cls._col_kind(dt) for col, dt in reference.dtypes.items()},
        )
        gate.ref_null_rates = reference.isna().mean().to_dict()

        full_shap = _BackendAdapter.compute_shap(model, reference)
        margin = np.sum(full_shap, axis=1)
        gate.margin_std = max(float(np.std(margin)), 1e-6)
        gate.ref_shap = full_shap / gate.margin_std
        gate.ref_margin = margin / gate.margin_std

        # Calibrate across a grid of sample sizes n using strictly DISJOINT splits
        n_total = len(reference)
        max_n = n_total // 2
        grid_n = [n for n in (250, 500, 1000, 2000, 4000) if n <= max_n]
        if max_n not in grid_n:
            grid_n.append(max_n)

        rng = np.random.default_rng(seed)
        med_records = {"mean": [], "max": [], "margin": []}
        T_distributions = []

        for n in grid_n:
            n_iter = n_bootstraps if n == grid_n[-1] else max(250, n_bootstraps // 3)
            s_mean = np.zeros(n_iter)
            s_max = np.zeros(n_iter)
            s_margin = np.zeros(n_iter)

            for b in range(n_iter):
                # Disjoint draw of 2*n prevents artificial overlap shrinkage
                idx = rng.choice(n_total, size=2 * n, replace=False)
                a_idx, b_idx = idx[:n], idx[n:]

                w1_vec = cls._vectorized_equal_w1(gate.ref_shap[a_idx], gate.ref_shap[b_idx])
                s_mean[b] = np.mean(w1_vec)
                s_max[b] = np.max(w1_vec)

                m_a = np.sort(gate.ref_margin[a_idx])
                m_b = np.sort(gate.ref_margin[b_idx])
                s_margin[b] = np.mean(np.abs(m_a - m_b))

            med_m, med_x, med_g = np.median(s_mean), np.median(s_max), np.median(s_margin)
            med_records["mean"].append(max(med_m, 1e-9))
            med_records["max"].append(max(med_x, 1e-9))
            med_records["margin"].append(max(med_g, 1e-9))

            # Pivotal joint statistic T = max_k(S_k / null_median_k)
            if n == grid_n[-1]:
                T_vals = np.maximum.reduce([
                    s_mean / max(med_m, 1e-9),
                    s_max / max(med_x, 1e-9),
                    s_margin / max(med_g, 1e-9),
                ])
                T_distributions.extend(T_vals)

        log_ns = np.log(grid_n)
        for key in ("mean", "max", "margin"):
            slope, intercept = np.polyfit(log_ns, np.log(med_records[key]), 1)
            # Theoretical 1D W1 null scaling is O(n^-0.5); clamp to safe bounds
            slope = float(np.clip(slope, -0.65, -0.35))
            gate.power_law_medians[key] = (float(intercept), slope)

        gate.tau_T_p99 = float(np.quantile(T_distributions, 1.0 - alpha))

        # Calibrate Null-Rate Delta Pre-Gate
        null_mat = reference.isna().to_numpy(dtype=float)
        n_cal = grid_n[-1]
        null_deltas = []
        for _ in range(500):
            idx = rng.choice(n_total, size=2 * n_cal, replace=False)
            delta = np.max(np.abs(null_mat[idx[:n_cal]].mean(axis=0) - null_mat[idx[n_cal:]].mean(axis=0)))
            null_deltas.append(delta)
        gate.max_null_delta_tau = max(float(np.quantile(null_deltas, 1.0 - alpha)), null_delta_floor)

        return gate

    def check(
        self,
        current: pd.DataFrame,
        min_samples: int = 250,
        max_consecutive_skips: int | None = None,
        state_file: str | Path | None = None,
    ) -> GateReport:
        # 1. Schema & Column Order/Kind Verification
        cur_cols = list(current.columns)
        if cur_cols != self.contract.features:
            missing = set(self.contract.features) - set(cur_cols)
            extra = set(cur_cols) - set(self.contract.features)
            if not missing and not extra:
                msg = f"Column order mismatch. Expected {self.contract.features[:3]}..., got {cur_cols[:3]}..."
            else:
                msg = f"Column mismatch. Missing: {missing}, Extra: {extra}"
            return GateReport("INVALID_SCHEMA", int(ExitCode.DATA_OR_SCHEMA_ERROR), 0, self.tau_T_p99, 0, 0, 0, 0, [], [msg])

        kind_errors = []
        for col, dt in current.dtypes.items():
            # Note: int columns promoted to float64 due to NaNs both map to 'numeric'
            if self._col_kind(dt) != self.contract.kinds[col]:
                kind_errors.append(f"{col}: expected {self.contract.kinds[col]}, got {self._col_kind(dt)} ({dt})")
        if kind_errors:
            return GateReport("INVALID_DTYPE", int(ExitCode.DATA_OR_SCHEMA_ERROR), 0, self.tau_T_p99, 0, 0, 0, 0, [], kind_errors)

        # 2. Sample Volume Guardrail -> Routes to Exit Code 20 (Never retrain on an outage)
        n_cur = len(current)
        if n_cur < min_samples:
            return GateReport(
                "INSUFFICIENT_VOLUME",
                int(ExitCode.DATA_OR_SCHEMA_ERROR),
                0.0,
                self.tau_T_p99,
                0.0,
                0.0,
                0.0,
                0,
                [],
                [f"Window size {n_cur} < min_samples {min_samples}. Suspect upstream outage."],
            )

        # 3. Full-Window Null-Rate Pre-Gate & SHAP Evaluation (No subsampling loss)
        cur_nulls = current.isna().mean().to_dict()
        cur_shap = _BackendAdapter.compute_shap(self.model, current) / self.margin_std
        cur_margin = np.sum(cur_shap, axis=1)

        # Compare full current window against full reference pool using exact 1D W1
        n_features = cur_shap.shape[1]
        w1_features = np.zeros(n_features)
        for j in range(n_features):
            w1_features[j] = wasserstein_distance(self.ref_shap[:, j], cur_shap[:, j])

        mean_w1 = float(np.mean(w1_features))
        max_w1 = float(np.max(w1_features))
        margin_w1 = float(wasserstein_distance(self.ref_margin, cur_margin))

        # Adjust expected null medians for unequal sample size (n_cur vs n_ref)
        # Effective two-sample variance scales with n_eff = (n_ref * n_cur) / (n_ref + n_cur) / 0.5
        n_ref = len(self.ref_shap)
        n_equiv = int((2.0 * n_ref * n_cur) / (n_ref + n_cur))
        med_mean, med_max, med_margin = self._predict_null_medians(n_equiv)

        joint_T = max(
            mean_w1 / max(med_mean, 1e-9),
            max_w1 / max(med_max, 1e-9),
            margin_w1 / max(med_margin, 1e-9),
        )

        # Feature Diagnostics
        ref_imp = np.mean(np.abs(self.ref_shap), axis=0)
        cur_imp = np.mean(np.abs(cur_shap), axis=0)
        ref_ranks = np.argsort(np.argsort(-ref_imp)) + 1
        cur_ranks = np.argsort(np.argsort(-cur_imp)) + 1

        drivers = []
        null_violations = []
        for j, col in enumerate(self.contract.features):
            null_delta = float(cur_nulls[col] - self.ref_null_rates[col])
            if abs(null_delta) > self.max_null_delta_tau:
                null_violations.append(f"{col} (ΔNaN: {null_delta:+.1%})")
            drivers.append({
                "feature": col,
                "w1_scaled": float(w1_features[j]),
                "rank_ref": int(ref_ranks[j]),
                "rank_cur": int(cur_ranks[j]),
                "null_delta": null_delta,
            })
        drivers.sort(key=lambda x: x["w1_scaled"], reverse=True)

        # 4. Data Quality Pre-Gate: Block retraining if NaNs spiked (broken upstream ETL)
        if null_violations:
            return GateReport(
                verdict="DATA_QUALITY_SUSPECT",
                exit_code=int(ExitCode.DATA_OR_SCHEMA_ERROR),
                joint_stat_T=joint_T,
                tau_T_p99=self.tau_T_p99,
                mean_w1=mean_w1,
                max_w1=max_w1,
                margin_w1=margin_w1,
                consecutive_skips=0,
                top_drivers=drivers,
                errors=[f"Null-rate spike exceeded {self.max_null_delta_tau:.1%}: {', '.join(null_violations)}"],
            )

        # 5. Drift Decision & State-File Staleness Override (Concept Drift Safety Floor)
        breached = joint_T > self.tau_T_p99
        skips = 0
        if state_file is not None:
            path = Path(state_file)
            if path.exists():
                skips = json.loads(path.read_text()).get("consecutive_skips", 0)

        if breached:
            verdict = "RETRAIN_REQUIRED"
            code = int(ExitCode.TRIGGER_RETRAIN)
            skips = 0
        elif max_consecutive_skips is not None and (skips + 1) >= max_consecutive_skips:
            verdict = "FORCED_RETRAIN_STALENESS"
            code = int(ExitCode.TRIGGER_RETRAIN)
            skips = 0
        else:
            verdict = "SKIP_RETRAIN"
            code = int(ExitCode.SKIP_RETRAIN)
            skips += 1

        if state_file is not None:
            Path(state_file).write_text(json.dumps({"consecutive_skips": skips}))

        return GateReport(
            verdict=verdict,
            exit_code=code,
            joint_stat_T=joint_T,
            tau_T_p99=self.tau_T_p99,
            mean_w1=mean_w1,
            max_w1=max_w1,
            margin_w1=margin_w1,
            consecutive_skips=skips,
            top_drivers=drivers,
            errors=[],
        )