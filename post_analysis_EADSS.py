import json
import os
import re
from typing import List, Optional, Tuple

import pandas as pd

# -------------------------
# CONFIG - edit here
# -------------------------
TEST_METRICS_PATH = "cv_outputs/test_session_metrics.csv"
QUESTIONNAIRES_XLSX = "resultados_questionarios_calculados.xlsx"
OUTDIR = "cv_outputs"
# -------------------------


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}

    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_map:
            return lower_map[key]

    for cand in candidates:
        key = cand.strip().lower()
        for c in cols:
            if key in str(c).strip().lower():
                return c
    return None


def parse_session_id(session_id: str) -> Tuple[str, str, str]:
    parts = str(session_id).split("|")
    participant = parts[0].strip() if len(parts) > 0 else ""
    sessnum = parts[1].strip() if len(parts) > 1 else ""
    module = parts[2].strip() if len(parts) > 2 else ""
    return participant, sessnum, module


def parse_sessnum_from_string(v: str) -> str:
    if pd.isna(v):
        return ""
    m = re.search(r"(\d+)", str(v))
    return m.group(1) if m else ""


def require_columns(df: pd.DataFrame, required: List[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def main() -> None:
    if not os.path.exists(TEST_METRICS_PATH):
        raise FileNotFoundError(f"Missing file: {TEST_METRICS_PATH}")
    if not os.path.exists(QUESTIONNAIRES_XLSX):
        raise FileNotFoundError(f"Missing file: {QUESTIONNAIRES_XLSX}")

    sess = pd.read_csv(TEST_METRICS_PATH)
    require_columns(
        sess,
        ["session_id", "participant_id", "mse_model", "mse_baseline", "improvement_pct"],
        TEST_METRICS_PATH,
    )

    parsed = sess["session_id"].astype(str).apply(parse_session_id)
    sess["participant"] = parsed.apply(lambda x: x[0])
    sess["sessnum"] = parsed.apply(lambda x: x[1])
    sess["module"] = parsed.apply(lambda x: x[2])

    q = pd.read_excel(QUESTIONNAIRES_XLSX)

    participant_col = find_column(
        q,
        [
            "Anonymized ID",
            "anonymized id",
            "anonymizedid",
            "participant",
            "participant_id",
        ],
    )
    session_col = find_column(
        q,
        ["Session ID", "session id", "sessionid", "session", "sessao", "sessÃ£o"],
    )

    if participant_col is None:
        raise RuntimeError("Could not detect participant column in questionnaire file.")
    if session_col is None:
        raise RuntimeError("Could not detect session column in questionnaire file.")

    required_eads = ["EADS_Depressao", "EADS_Ansiedade", "EADS_Stress"]
    require_columns(q, required_eads, QUESTIONNAIRES_XLSX)

    q_norm = q.copy()
    q_norm["participant"] = q_norm[participant_col].astype(str).str.strip()
    q_norm["sessnum"] = q_norm[session_col].apply(parse_sessnum_from_string).astype(str).str.strip()

    for col in required_eads:
        q_norm[col] = pd.to_numeric(q_norm[col], errors="coerce")

    q_eads = q_norm.dropna(subset=required_eads, how="all").copy()

    q_eads_dedup = (
        q_eads.groupby(["participant", "sessnum"], as_index=False)[required_eads]
        .mean()
        .reset_index(drop=True)
    )

    merged = sess.merge(
        q_eads_dedup,
        how="left",
        on=["participant", "sessnum"],
    )
    merged["has_eads"] = merged[required_eads].notna().all(axis=1)

    part_df = (
        merged.groupby("participant", as_index=False)
        .agg(
            n_sessions=("session_id", "count"),
            n_sessions_with_eads=("has_eads", "sum"),
            mse_model_mean=("mse_model", "mean"),
            mse_baseline_mean=("mse_baseline", "mean"),
            improvement_pct_mean=("improvement_pct", "mean"),
            EADS_Depressao_mean=("EADS_Depressao", "mean"),
            EADS_Ansiedade_mean=("EADS_Ansiedade", "mean"),
            EADS_Stress_mean=("EADS_Stress", "mean"),
        )
        .sort_values("participant")
        .reset_index(drop=True)
    )

    n_total_sessions = int(len(merged))
    n_matched_sessions = int(merged["has_eads"].sum())
    n_total_participants = int(part_df["participant"].nunique())
    n_participants_with_eads = int((part_df["n_sessions_with_eads"] > 0).sum())

    summary = {
        "test_metrics_path": TEST_METRICS_PATH,
        "questionnaires_path": QUESTIONNAIRES_XLSX,
        "n_sessions_test": n_total_sessions,
        "n_sessions_with_eads": n_matched_sessions,
        "match_rate_sessions_pct": float(100.0 * n_matched_sessions / n_total_sessions)
        if n_total_sessions > 0
        else 0.0,
        "n_participants_test": n_total_participants,
        "n_participants_with_any_eads": n_participants_with_eads,
        "participant_col_used": str(participant_col),
        "session_col_used": str(session_col),
        "spearman_analysis_removed": True,
    }

    os.makedirs(OUTDIR, exist_ok=True)
    out_merged = os.path.join(OUTDIR, "eads_merged_test_sessions.csv")
    out_participant = os.path.join(OUTDIR, "eads_participant_level.csv")
    out_summary = os.path.join(OUTDIR, "eads_link_summary.json")
    out_corr_legacy = os.path.join(OUTDIR, "eads_corr_participant_level.csv")

    merged.to_csv(out_merged, index=False)
    part_df.to_csv(out_participant, index=False)
    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if os.path.exists(out_corr_legacy):
        os.remove(out_corr_legacy)

    print("EADS post-analysis complete.")
    print(f" - sessions matched with EADS: {n_matched_sessions}/{n_total_sessions}")
    print(f" - participants with any EADS: {n_participants_with_eads}/{n_total_participants}")
    print(f" - removed legacy Spearman file if present: {out_corr_legacy}")
    print(f"Saved outputs:\n - {out_merged}\n - {out_participant}\n - {out_summary}")


if __name__ == "__main__":
    main()
