import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -------------------------
# CONFIG - edit here
# -------------------------
TEST_METRICS_PATH = "cv_outputs/test_session_metrics.csv"
RUN_SUMMARY_PATH = "cv_outputs/run_summary.json"
LENGTH_BIN_PATH = "cv_outputs/test_metrics_by_length_bin.csv"
FEATURE_METRICS_PATH = "cv_outputs/feature_reconstruction_metrics.csv"
OUTDIR = "cv_outputs"
# -------------------------


def require_columns(df: pd.DataFrame, required: List[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def parse_session_id(session_id: str) -> Tuple[str, str, str]:
    parts = str(session_id).split("|")
    participant = parts[0].strip() if len(parts) > 0 else ""
    sessnum = parts[1].strip() if len(parts) > 1 else ""
    module = parts[2].strip() if len(parts) > 2 else ""
    return participant, sessnum, module


def to_native(v):
    if isinstance(v, (np.integer, np.int64, np.int32)):
        return int(v)
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    return v


def frame_to_records(df: pd.DataFrame) -> List[Dict]:
    out = []
    for rec in df.to_dict(orient="records"):
        out.append({k: to_native(v) for k, v in rec.items()})
    return out


def main() -> None:
    if not os.path.exists(TEST_METRICS_PATH):
        raise FileNotFoundError(f"test metrics not found: {TEST_METRICS_PATH}")

    df = pd.read_csv(TEST_METRICS_PATH)
    require_columns(
        df,
        [
            "session_id",
            "participant_id",
            "mse_model",
            "mse_baseline",
            "improvement_pct",
            "length",
            "length_raw",
        ],
        TEST_METRICS_PATH,
    )

    parsed = df["session_id"].astype(str).apply(parse_session_id)
    df["participant_from_session"] = parsed.apply(lambda x: x[0])
    df["session_number"] = parsed.apply(lambda x: x[1])
    df["module_id_from_session"] = parsed.apply(lambda x: x[2])

    mismatch_n = int((df["participant_id"].astype(str) != df["participant_from_session"]).sum())
    if mismatch_n > 0:
        print(
            f"[warn] participant mismatch in {mismatch_n} rows between participant_id and parsed session_id."
        )

    df["improved"] = df["mse_model"] < df["mse_baseline"]

    by_participant = (
        df.groupby("participant_id", as_index=False)
        .agg(
            n_sessions=("session_id", "count"),
            mean_mse_model=("mse_model", "mean"),
            mean_mse_baseline=("mse_baseline", "mean"),
            mean_improvement_pct=("improvement_pct", "mean"),
            median_improvement_pct=("improvement_pct", "median"),
            mean_length=("length", "mean"),
            mean_length_raw=("length_raw", "mean"),
            pct_sessions_improved=("improved", "mean"),
        )
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    by_participant["pct_sessions_improved"] = 100.0 * by_participant["pct_sessions_improved"]

    by_module = (
        df.groupby("module_id_from_session", as_index=False)
        .agg(
            n_sessions=("session_id", "count"),
            mean_mse_model=("mse_model", "mean"),
            mean_mse_baseline=("mse_baseline", "mean"),
            mean_improvement_pct=("improvement_pct", "mean"),
            median_improvement_pct=("improvement_pct", "median"),
            pct_sessions_improved=("improved", "mean"),
            mean_length_raw=("length_raw", "mean"),
        )
        .sort_values("module_id_from_session")
        .reset_index(drop=True)
    )
    by_module["pct_sessions_improved"] = 100.0 * by_module["pct_sessions_improved"]

    run_summary = {}
    if os.path.exists(RUN_SUMMARY_PATH):
        with open(RUN_SUMMARY_PATH, "r", encoding="utf-8") as fh:
            run_summary = json.load(fh)

    length_bins = []
    if os.path.exists(LENGTH_BIN_PATH):
        bins_df = pd.read_csv(LENGTH_BIN_PATH)
        length_bins = frame_to_records(bins_df)

    feature_top5 = []
    if os.path.exists(FEATURE_METRICS_PATH):
        feat_df = pd.read_csv(FEATURE_METRICS_PATH)
        if "nrmse_by_std" in feat_df.columns:
            feat_df = feat_df.sort_values(
                by="nrmse_by_std", ascending=False, na_position="last"
            ).reset_index(drop=True)
        feature_top5 = frame_to_records(feat_df.head(5))

    summary = {
        "n_sessions_test": int(len(df)),
        "n_participants_test": int(df["participant_id"].nunique()),
        "mean_mse_model_session_level": float(df["mse_model"].mean()),
        "mean_mse_baseline_session_level": float(df["mse_baseline"].mean()),
        "mean_improvement_pct_session_level": float(df["improvement_pct"].mean()),
        "median_improvement_pct_session_level": float(df["improvement_pct"].median()),
        "pct_sessions_improved": float(100.0 * df["improved"].mean()),
        "mean_mse_model_participant_level": float(by_participant["mean_mse_model"].mean()),
        "mean_mse_baseline_participant_level": float(by_participant["mean_mse_baseline"].mean()),
        "mean_improvement_pct_participant_level": float(
            by_participant["mean_improvement_pct"].mean()
        ),
        "participant_id_mismatch_count": int(mismatch_n),
        "run_summary_snapshot": {
            "best_mean_val_loss": run_summary.get("best_mean_val_loss"),
            "best_std_val_loss": run_summary.get("best_std_val_loss"),
            "epochs_final": run_summary.get("epochs_final"),
            "test_mean_improvement_pct": run_summary.get("test_mean_improvement_pct"),
            "maxlen_used": run_summary.get("maxlen_used"),
            "percent_sessions_truncated_train": run_summary.get("percent_sessions_truncated"),
            "percent_steps_removed_train": run_summary.get("percent_steps_removed"),
        },
        "length_bin_summary": length_bins,
        "feature_reconstruction_top5_nrmse": feature_top5,
    }

    os.makedirs(OUTDIR, exist_ok=True)
    out_summary = os.path.join(OUTDIR, "post_analysis_summary.json")
    out_participant = os.path.join(OUTDIR, "post_analysis_by_participant.csv")
    out_module = os.path.join(OUTDIR, "post_analysis_by_module.csv")

    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    by_participant.to_csv(out_participant, index=False)
    by_module.to_csv(out_module, index=False)

    print("Post-analysis summary complete.")
    print(f" - n_sessions_test: {summary['n_sessions_test']}")
    print(f" - n_participants_test: {summary['n_participants_test']}")
    print(
        f" - session-level mean improvement: {summary['mean_improvement_pct_session_level']:.3f}%"
    )
    print(
        f" - participant-level mean improvement: {summary['mean_improvement_pct_participant_level']:.3f}%"
    )
    print(f"Saved outputs:\n - {out_summary}\n - {out_participant}\n - {out_module}")


if __name__ == "__main__":
    main()
