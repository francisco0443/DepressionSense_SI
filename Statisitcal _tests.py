import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# -------------------------
# CONFIG - edit here
# -------------------------
TEST_METRICS_PATH = "cv_outputs/test_session_metrics.csv"
OUTDIR = "cv_outputs"
ALTERNATIVE = "two-sided"  # "two-sided" (conservative) 
# -------------------------


def require_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def main() -> None:
    if not os.path.exists(TEST_METRICS_PATH):
        raise FileNotFoundError(f"File not found: {TEST_METRICS_PATH}")

    df = pd.read_csv(TEST_METRICS_PATH)
    require_columns(
        df,
        ["participant_id", "session_id", "mse_model", "mse_baseline", "improvement_pct"],
        TEST_METRICS_PATH,
    )

    # Independence-aware aggregation: one paired value per participant.
    per_participant = (
        df.groupby("participant_id", as_index=False)
        .agg(
            n_sessions=("session_id", "count"),
            mse_model=("mse_model", "mean"),
            mse_baseline=("mse_baseline", "mean"),
            improvement_pct_mean=("improvement_pct", "mean"),
        )
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    per_participant["diff_baseline_minus_model"] = (
        per_participant["mse_baseline"] - per_participant["mse_model"]
    )

    diffs = per_participant["diff_baseline_minus_model"].to_numpy(dtype=float)
    if len(diffs) < 5:
        raise RuntimeError("Too few participants for a stable paired test.")

    # Single inferential test: Wilcoxon signed-rank (paired, non-parametric).
    test_res = wilcoxon(diffs, alternative=ALTERNATIVE)

    summary = {
        "test_name": "Wilcoxon signed-rank (paired, participant-level)",
        "alternative": ALTERNATIVE,
        "n_participants": int(len(per_participant)),
        "mean_mse_model": float(per_participant["mse_model"].mean()),
        "mean_mse_baseline": float(per_participant["mse_baseline"].mean()),
        "mean_diff_baseline_minus_model": float(np.mean(diffs)),
        "median_diff_baseline_minus_model": float(np.median(diffs)),
        "improved_participants_pct": float(100.0 * np.mean(diffs > 0)),
        "wilcoxon_statistic": float(test_res.statistic),
        "p_value": float(test_res.pvalue),
    }

    os.makedirs(OUTDIR, exist_ok=True)
    out_summary = os.path.join(OUTDIR, "statistical_test_summary.json")
    out_participants = os.path.join(OUTDIR, "participant_level_metrics_for_test.csv")

    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    per_participant.to_csv(out_participants, index=False)

    print("Statistical test complete.")
    print(f" - test: {summary['test_name']}")
    print(f" - n_participants: {summary['n_participants']}")
    print(f" - mean_mse_model: {summary['mean_mse_model']:.6f}")
    print(f" - mean_mse_baseline: {summary['mean_mse_baseline']:.6f}")
    print(f" - p_value: {summary['p_value']:.6g}")
    print(f"Saved outputs:\n - {out_summary}\n - {out_participants}")


if __name__ == "__main__":
    main()
