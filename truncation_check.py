import json
import os
from typing import Dict, List

import pandas as pd

# -------------------------
# CONFIG - edit here
# -------------------------
RUN_SUMMARY_PATH = "cv_outputs/run_summary.json"
TEST_METRICS_PATH = "cv_outputs/test_session_metrics.csv"
OUTDIR = "cv_outputs"
TOP_N = 10
# -------------------------


def require_columns(df: pd.DataFrame, required: List[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def to_records(df: pd.DataFrame) -> List[Dict]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(bool)
    return out.to_dict(orient="records")


def main() -> None:
    if not os.path.exists(RUN_SUMMARY_PATH):
        raise FileNotFoundError(f"run_summary not found: {RUN_SUMMARY_PATH}")
    if not os.path.exists(TEST_METRICS_PATH):
        raise FileNotFoundError(f"test metrics not found: {TEST_METRICS_PATH}")

    with open(RUN_SUMMARY_PATH, "r", encoding="utf-8") as fh:
        run_summary = json.load(fh)
    if "maxlen_used" not in run_summary:
        raise ValueError("run_summary.json must contain 'maxlen_used'.")
    maxlen_used = int(run_summary["maxlen_used"])

    df = pd.read_csv(TEST_METRICS_PATH)
    require_columns(
        df,
        ["participant_id", "mse_model", "mse_baseline", "improvement_pct", "length_raw"],
        TEST_METRICS_PATH,
    )

    df = df.copy()
    df["length_raw"] = pd.to_numeric(df["length_raw"], errors="coerce")
    if df["length_raw"].isna().any():
        raise ValueError("Found NaN in length_raw after numeric conversion.")

    df["is_truncated"] = df["length_raw"] > maxlen_used
    df["steps_removed"] = (df["length_raw"] - maxlen_used).clip(lower=0).astype(int)

    session_group = (
        df.groupby("is_truncated", observed=False)
        .agg(
            count=("improvement_pct", "size"),
            mean_improvement_pct=("improvement_pct", "mean"),
            median_improvement_pct=("improvement_pct", "median"),
            std_improvement_pct=("improvement_pct", "std"),
            mean_mse_model=("mse_model", "mean"),
            mean_mse_baseline=("mse_baseline", "mean"),
            total_steps_removed=("steps_removed", "sum"),
        )
        .reset_index()
    )
    session_group["group"] = session_group["is_truncated"].map(
        {False: "not_truncated", True: "truncated"}
    )
    session_group = session_group[
        [
            "group",
            "count",
            "mean_improvement_pct",
            "median_improvement_pct",
            "std_improvement_pct",
            "mean_mse_model",
            "mean_mse_baseline",
            "total_steps_removed",
        ]
    ]

    part_base = (
        df.groupby("participant_id", observed=False)
        .agg(
            n_sessions=("improvement_pct", "size"),
            trunc_sessions=("is_truncated", "sum"),
            steps_removed=("steps_removed", "sum"),
            mean_improvement_pct=("improvement_pct", "mean"),
            median_improvement_pct=("improvement_pct", "median"),
        )
        .reset_index()
    )
    part_base["trunc_rate"] = part_base["trunc_sessions"] / part_base["n_sessions"]
    part_base["has_truncation"] = part_base["trunc_sessions"] > 0

    part_tr = (
        df[df["is_truncated"]]
        .groupby("participant_id", observed=False)["improvement_pct"]
        .mean()
        .rename("mean_improvement_truncated")
    )
    part_nt = (
        df[~df["is_truncated"]]
        .groupby("participant_id", observed=False)["improvement_pct"]
        .mean()
        .rename("mean_improvement_not_truncated")
    )
    participant_stats = part_base.merge(part_tr, on="participant_id", how="left").merge(
        part_nt, on="participant_id", how="left"
    )
    participant_stats["delta_trunc_minus_not_trunc"] = (
        participant_stats["mean_improvement_truncated"]
        - participant_stats["mean_improvement_not_truncated"]
    )
    participant_stats = participant_stats.sort_values(
        ["steps_removed", "trunc_sessions", "participant_id"], ascending=[False, False, True]
    ).reset_index(drop=True)

    participant_group = (
        participant_stats.groupby("has_truncation", observed=False)
        .agg(
            n_participants=("participant_id", "size"),
            mean_participant_improvement_pct=("mean_improvement_pct", "mean"),
            median_participant_improvement_pct=("mean_improvement_pct", "median"),
            mean_participant_trunc_rate=("trunc_rate", "mean"),
        )
        .reset_index()
    )
    participant_group["group"] = participant_group["has_truncation"].map(
        {False: "participants_without_truncation", True: "participants_with_truncation"}
    )
    participant_group = participant_group[
        [
            "group",
            "n_participants",
            "mean_participant_improvement_pct",
            "median_participant_improvement_pct",
            "mean_participant_trunc_rate",
        ]
    ]

    n_sessions = int(len(df))
    n_sessions_truncated = int(df["is_truncated"].sum())
    n_steps_total = int(df["length_raw"].sum())
    n_steps_removed_total = int(df["steps_removed"].sum())

    summary = {
        "maxlen_used": int(maxlen_used),
        "n_sessions": n_sessions,
        "n_sessions_truncated": n_sessions_truncated,
        "percent_sessions_truncated": (
            100.0 * float(n_sessions_truncated) / float(n_sessions) if n_sessions > 0 else 0.0
        ),
        "n_steps_total_raw": n_steps_total,
        "n_steps_removed_total": n_steps_removed_total,
        "percent_steps_removed_vs_raw": (
            100.0 * float(n_steps_removed_total) / float(n_steps_total) if n_steps_total > 0 else 0.0
        ),
        "session_level_summary": to_records(session_group),
        "participant_level_group_summary": to_records(participant_group),
        "top_participants_by_steps_removed": to_records(
            participant_stats.head(int(TOP_N))[
                [
                    "participant_id",
                    "n_sessions",
                    "trunc_sessions",
                    "trunc_rate",
                    "steps_removed",
                    "mean_improvement_pct",
                    "mean_improvement_truncated",
                    "mean_improvement_not_truncated",
                    "delta_trunc_minus_not_trunc",
                ]
            ]
        ),
    }

    os.makedirs(OUTDIR, exist_ok=True)
    out_summary_json = os.path.join(OUTDIR, "truncation_check_summary.json")
    out_session_csv = os.path.join(OUTDIR, "truncation_check_session_groups.csv")
    out_participant_csv = os.path.join(OUTDIR, "truncation_check_by_participant.csv")
    out_participant_group_csv = os.path.join(OUTDIR, "truncation_check_participant_groups.csv")

    with open(out_summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    session_group.to_csv(out_session_csv, index=False)
    participant_stats.to_csv(out_participant_csv, index=False)
    participant_group.to_csv(out_participant_group_csv, index=False)

    print("Truncation check complete.")
    print(f" - maxlen_used: {maxlen_used}")
    print(f" - n_sessions_truncated: {n_sessions_truncated}/{n_sessions}")
    print(f" - n_steps_removed_total: {n_steps_removed_total}/{n_steps_total}")
    print(f"Saved outputs:\n - {out_summary_json}\n - {out_session_csv}\n - {out_participant_csv}\n - {out_participant_group_csv}")


if __name__ == "__main__":
    main()
