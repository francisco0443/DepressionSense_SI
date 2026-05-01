import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import LSTM_GroupK_CV as cv

# -------------------------
# CONFIG - edit here
# -------------------------
FEATURES_CSV = "action_sequences_lstm_features.csv"
QUESTIONNAIRES_XLSX = "resultados_questionarios_calculados.xlsx"
METADATA_PATH = "cv_outputs/metadata.json"
WEIGHTS_PATH = "cv_outputs/best_model_final.weights.h5"
PARTICIPANTS_TRAIN_TXT = "cv_outputs/participants_train.txt"
OUTDIR = "cv_outputs"
LOW_QUANTILE = 0.25
HIGH_QUANTILE = 0.75
MULTIPLY_EADS_BY_2 = True
ALTERNATIVE = "two-sided"
BATCH_SIZE_PRED = 32

BEHAVIOR_FEATURES = [
    "cursor_speed",
    "acceleration_mean",
    "jitter",
    "direction_changes",
    "curvature_mean",
    "rate_of_curvature",
    "distance_travelled",
    "movement_offset",
    "straightness",
    "self_intersections",
    "action_duration",
    "time_since_last_action",
]
RECON_NRMSE_PREFIX = "recon_nrmse__"
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


def parse_sessnum_from_string(v: str) -> str:
    if pd.isna(v):
        return ""
    m = re.search(r"(\d+)", str(v))
    return m.group(1) if m else ""


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


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    out = np.empty_like(q)
    out[order] = q
    return out


def rank_biserial_from_u(u_stat: float, n_low: int, n_high: int) -> float:
    denom = float(n_low * n_high)
    if denom <= 0:
        return np.nan
    return float(1.0 - 2.0 * (u_stat / denom))


def read_participants_txt(path: str) -> Set[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing participant list: {path}")
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            v = line.strip()
            if v:
                out.add(v)
    return out


def run_mwu_high_low(
    part_df: pd.DataFrame,
    eads_cols: List[str],
    feature_cols: List[str],
    analysis_label: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    rows: List[Dict] = []
    subgroup_meta: Dict[str, Dict] = {}

    for target in eads_cols:
        temp = part_df.dropna(subset=[target]).copy()
        if temp.empty:
            continue

        q_low = float(temp[target].quantile(LOW_QUANTILE))
        q_high = float(temp[target].quantile(HIGH_QUANTILE))
        low = temp[temp[target] <= q_low].copy()
        high = temp[temp[target] >= q_high].copy()
        n_low_group = int(len(low))
        n_high_group = int(len(high))

        subgroup_meta[target] = {
            "n_available": int(len(temp)),
            "q_low_value": q_low,
            "q_high_value": q_high,
            "n_low_group": n_low_group,
            "n_high_group": n_high_group,
            "low_quantile": float(LOW_QUANTILE),
            "high_quantile": float(HIGH_QUANTILE),
        }

        if n_low_group < 3 or n_high_group < 3:
            for feat_name in feature_cols:
                rows.append(
                    {
                        "analysis": analysis_label,
                        "target": target,
                        "feature": feat_name,
                        "n_low": n_low_group,
                        "n_high": n_high_group,
                        "mean_low": np.nan,
                        "mean_high": np.nan,
                        "median_low": np.nan,
                        "median_high": np.nan,
                        "u_statistic": np.nan,
                        "p_value": np.nan,
                        "rank_biserial": np.nan,
                    }
                )
            continue

        for feat_name in feature_cols:
            x = low[feat_name].dropna().to_numpy(dtype=float)
            y = high[feat_name].dropna().to_numpy(dtype=float)

            if len(x) < 3 or len(y) < 3:
                u_stat = np.nan
                p_val = np.nan
                rbc = np.nan
            else:
                test = mannwhitneyu(x, y, alternative=ALTERNATIVE, method="asymptotic")
                u_stat = float(test.statistic)
                p_val = float(test.pvalue)
                rbc = rank_biserial_from_u(u_stat, len(x), len(y))

            rows.append(
                {
                    "analysis": analysis_label,
                    "target": target,
                    "feature": feat_name,
                    "n_low": int(len(x)),
                    "n_high": int(len(y)),
                    "mean_low": float(np.mean(x)) if len(x) else np.nan,
                    "mean_high": float(np.mean(y)) if len(y) else np.nan,
                    "median_low": float(np.median(x)) if len(x) else np.nan,
                    "median_high": float(np.median(y)) if len(y) else np.nan,
                    "u_statistic": u_stat,
                    "p_value": p_val,
                    "rank_biserial": rbc,
                }
            )

    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_adj_fdr_bh"] = np.nan
        for target in res["target"].dropna().unique():
            mask = (res["target"] == target) & res["p_value"].notna()
            if mask.any():
                p_adj = fdr_bh(res.loc[mask, "p_value"].to_numpy(dtype=float))
                res.loc[mask, "p_adj_fdr_bh"] = p_adj
        res["significant_fdr_0p05"] = res["p_adj_fdr_bh"] < 0.05
        res = res.sort_values(
            by=["target", "p_adj_fdr_bh", "p_value", "feature"],
            ascending=[True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)

    return res, subgroup_meta


def build_behavior_profiles(features_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feat = pd.read_csv(features_csv, sep=";")
    require_columns(feat, ["session_key"] + BEHAVIOR_FEATURES, features_csv)

    parsed = feat["session_key"].astype(str).apply(parse_session_id)
    feat["participant"] = parsed.apply(lambda x: x[0])
    feat["sessnum"] = parsed.apply(lambda x: x[1])

    session_feature = (
        feat.groupby(["participant", "sessnum"], as_index=False)[BEHAVIOR_FEATURES]
        .mean()
        .reset_index(drop=True)
    )
    participant_feature = (
        session_feature.groupby("participant", as_index=False)[BEHAVIOR_FEATURES]
        .mean()
        .reset_index(drop=True)
    )
    return feat, session_feature, participant_feature


def build_participant_eads(questionnaires_xlsx: str) -> Tuple[pd.DataFrame, str, str]:
    q = pd.read_excel(questionnaires_xlsx)
    participant_col = find_column(
        q,
        ["Anonymized ID", "anonymized id", "anonymizedid", "participant", "participant_id"],
    )
    session_col = find_column(
        q,
        ["Session ID", "session id", "sessionid", "session", "sessao", "sessÃ£o"],
    )

    if participant_col is None:
        raise RuntimeError("Could not detect participant column in questionnaire file.")
    if session_col is None:
        raise RuntimeError("Could not detect session column in questionnaire file.")

    eads_cols = ["EADS_Depressao", "EADS_Ansiedade", "EADS_Stress"]
    require_columns(q, eads_cols, questionnaires_xlsx)

    q_norm = q.copy()
    q_norm["participant"] = q_norm[participant_col].astype(str).str.strip()
    q_norm["sessnum"] = q_norm[session_col].apply(parse_sessnum_from_string).astype(str).str.strip()

    for c in eads_cols:
        q_norm[c] = pd.to_numeric(q_norm[c], errors="coerce")
        if MULTIPLY_EADS_BY_2:
            q_norm[c] = q_norm[c] * 2.0

    q_eads = q_norm.dropna(subset=eads_cols, how="all").copy()
    q_eads_session = (
        q_eads.groupby(["participant", "sessnum"], as_index=False)[eads_cols]
        .mean()
        .reset_index(drop=True)
    )
    participant_eads = (
        q_eads_session.groupby("participant", as_index=False)[eads_cols]
        .mean()
        .reset_index(drop=True)
    )
    return participant_eads, str(participant_col), str(session_col)


def build_reconstruction_nrmse_profiles(
    features_df_raw: pd.DataFrame,
    metadata: Dict,
    preprocessing_artifacts: Dict,
    weights_path: str,
    train_participants: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    session_col = str(metadata["session_col"])
    group_col = str(metadata["group_col"])
    timesteps = int(metadata["timesteps"])
    categorical_features = list(metadata["categorical_features"])
    continuous_features = list(metadata["continuous_features"])
    embedding_cfg = {
        k: (int(v[0]), int(v[1])) for k, v in metadata["embedding_config"].items()
    }
    best_params = metadata["best_params"]

    cv.DROP_XY_MODEL = bool(metadata.get("drop_xy_model", cv.DROP_XY_MODEL))

    df_for_model = features_df_raw.copy()
    if group_col not in df_for_model.columns:
        if session_col not in df_for_model.columns:
            raise ValueError(
                f"Missing both group_col='{group_col}' and session_col='{session_col}' "
                "needed to recover participant ids."
            )
        df_for_model[group_col] = (
            df_for_model[session_col].astype(str).apply(lambda s: parse_session_id(s)[0])
        )

    proc = cv.preprocess_transform_only(
        df_for_model,
        cat_maps=preprocessing_artifacts["cat_maps"],
        scalers=preprocessing_artifacts["scalers"],
        clip_params=preprocessing_artifacts["clip_params"],
    )

    X_cont, X_cat, _, cont_features_got, session_ids, participant_ids = cv.prepare_sequences(
        proc,
        session_col=session_col,
        group_col=group_col,
        categorical_features=categorical_features,
        maxlen=timesteps,
        continuous_features_ref=continuous_features,
    )
    if list(cont_features_got) != list(continuous_features):
        raise RuntimeError("Continuous feature order mismatch during reconstruction prep.")

    model = cv.build_lstm_autoencoder(
        timesteps=timesteps,
        num_cont_features=len(continuous_features),
        categorical_features_list=categorical_features,
        embedding_cfg=embedding_cfg,
        lstm_units=int(best_params["lstm_units"]),
        bottleneck_dim=int(best_params["bottleneck_dim"]),
        dropout_rate=float(best_params["dropout_rate"]),
    )
    model.load_weights(weights_path)

    X_cat_list = [X_cat[f] for f in categorical_features]
    pred = model.predict([X_cont] + X_cat_list, batch_size=BATCH_SIZE_PRED, verbose=0)
    mask = cv.make_mask(X_cat, "action_type").astype(bool)

    participants_arr = np.array(participant_ids, dtype=str)
    train_session_mask = np.array([p in train_participants for p in participants_arr], dtype=bool)

    n_features = len(continuous_features)
    std_ref = np.full((n_features,), np.nan, dtype=np.float64)
    ref_rows: List[Dict] = []

    for j, feat in enumerate(continuous_features):
        yt_train = X_cont[train_session_mask, :, j][mask[train_session_mask]].astype(np.float64)
        if yt_train.size == 0:
            yt_train = X_cont[:, :, j][mask].astype(np.float64)
        ref_std = float(np.std(yt_train)) if yt_train.size else np.nan
        ref_mean = float(np.mean(yt_train)) if yt_train.size else np.nan
        std_ref[j] = ref_std
        ref_rows.append(
            {
                "feature": feat,
                "reference_set": "train_sessions" if train_session_mask.any() else "all_sessions_fallback",
                "reference_n_points": int(yt_train.size),
                "reference_mean_true": ref_mean,
                "reference_std_true": ref_std,
            }
        )

    unique_participants = sorted(set(participant_ids))
    part_rows: List[Dict] = []
    for pid in unique_participants:
        idx = participants_arr == pid
        valid = mask[idx]
        rec: Dict[str, float] = {
            "participant": str(pid),
            "n_sessions_for_recon": int(np.sum(idx)),
        }
        for j, feat in enumerate(continuous_features):
            se_sum = float((((X_cont[idx, :, j] - pred[idx, :, j]) ** 2) * valid).sum())
            cnt = float(valid.sum())
            if cnt <= 0:
                nrmse = np.nan
                rmse = np.nan
            else:
                rmse = float(np.sqrt(se_sum / cnt))
                sref = std_ref[j]
                nrmse = float(rmse / sref) if np.isfinite(sref) and sref > 1e-12 else np.nan
            rec[f"{RECON_NRMSE_PREFIX}{feat}"] = nrmse
        part_rows.append(rec)

    session_rows: List[Dict] = []
    for i, sid in enumerate(session_ids):
        pid, sessnum, _ = parse_session_id(sid)
        valid_i = mask[i]
        rec = {
            "session_id": sid,
            "participant": pid,
            "sessnum": sessnum,
            "n_valid_timesteps": int(valid_i.sum()),
        }
        for j, feat in enumerate(continuous_features):
            if valid_i.sum() <= 0:
                nrmse = np.nan
            else:
                rmse = float(np.sqrt(np.mean((X_cont[i, valid_i, j] - pred[i, valid_i, j]) ** 2)))
                sref = std_ref[j]
                nrmse = float(rmse / sref) if np.isfinite(sref) and sref > 1e-12 else np.nan
            rec[f"{RECON_NRMSE_PREFIX}{feat}"] = nrmse
        session_rows.append(rec)

    part_recon = pd.DataFrame(part_rows).sort_values("participant").reset_index(drop=True)
    sess_recon = pd.DataFrame(session_rows).sort_values(["participant", "sessnum", "session_id"]).reset_index(drop=True)
    ref_df = pd.DataFrame(ref_rows).sort_values("feature").reset_index(drop=True)
    recon_feature_cols = [f"{RECON_NRMSE_PREFIX}{f}" for f in continuous_features]

    return part_recon, sess_recon, ref_df, recon_feature_cols


def main() -> None:
    required_paths = [
        FEATURES_CSV,
        QUESTIONNAIRES_XLSX,
        METADATA_PATH,
        WEIGHTS_PATH,
        PARTICIPANTS_TRAIN_TXT,
    ]
    for p in required_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    artifacts_rel = metadata.get("preprocessing_artifacts_path", "train_preprocessing_artifacts.joblib")
    artifacts_path = artifacts_rel
    if not os.path.isabs(artifacts_path):
        artifacts_path = os.path.join(os.path.dirname(METADATA_PATH), artifacts_rel)
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Missing preprocessing artifacts: {artifacts_path}")

    preprocessing_artifacts = joblib.load(artifacts_path)
    train_participants = read_participants_txt(PARTICIPANTS_TRAIN_TXT)

    feat_raw, _, participant_feature = build_behavior_profiles(FEATURES_CSV)
    participant_eads, participant_col_used, session_col_used = build_participant_eads(QUESTIONNAIRES_XLSX)

    part_recon, sess_recon, recon_ref_df, recon_feature_cols = build_reconstruction_nrmse_profiles(
        features_df_raw=feat_raw,
        metadata=metadata,
        preprocessing_artifacts=preprocessing_artifacts,
        weights_path=WEIGHTS_PATH,
        train_participants=train_participants,
    )

    eads_cols = ["EADS_Depressao", "EADS_Ansiedade", "EADS_Stress"]
    part_raw = participant_feature.merge(participant_eads, on="participant", how="inner")
    part_recon_mwu = part_recon.merge(participant_eads, on="participant", how="inner")
    if part_raw.empty or part_recon_mwu.empty:
        raise RuntimeError("No overlap between features and participant EADS.")

    raw_res, raw_subgroups = run_mwu_high_low(
        part_df=part_raw,
        eads_cols=eads_cols,
        feature_cols=BEHAVIOR_FEATURES,
        analysis_label="raw_behavior_features",
    )
    recon_res, recon_subgroups = run_mwu_high_low(
        part_df=part_recon_mwu,
        eads_cols=eads_cols,
        feature_cols=recon_feature_cols,
        analysis_label="reconstruction_nrmse_features",
    )

    summary = {
        "analysis_type": "exploratory_mwu_high_vs_low_all_participants",
        "features_source": FEATURES_CSV,
        "questionnaires_source": QUESTIONNAIRES_XLSX,
        "metadata_source": METADATA_PATH,
        "weights_source": WEIGHTS_PATH,
        "preprocessing_artifacts_source": artifacts_path,
        "n_participants_merged": int(part_raw["participant"].nunique()),
        "behavior_features_n": int(len(BEHAVIOR_FEATURES)),
        "behavior_features": BEHAVIOR_FEATURES,
        "eads_scaled_x2": bool(MULTIPLY_EADS_BY_2),
        "alternative": ALTERNATIVE,
        "subgroup_definition": {
            "low_quantile": float(LOW_QUANTILE),
            "high_quantile": float(HIGH_QUANTILE),
        },
        "participant_col_used": participant_col_used,
        "session_col_used": session_col_used,
        "raw_feature_analysis": {
            "n_tests_total": int(len(raw_res)),
            "n_significant_fdr_0p05_total": int(raw_res["significant_fdr_0p05"].sum()) if not raw_res.empty else 0,
            "subgroup_sizes_by_target": raw_subgroups,
            "feature_columns": BEHAVIOR_FEATURES,
        },
        "reconstruction_nrmse_analysis": {
            "definition": "Participant-level reconstruction nRMSE per feature from the trained LSTM autoencoder (mask-aware; non-padded timesteps only).",
            "normalization_reference": "std of true feature values on train participants (fallback all sessions if needed)",
            "n_tests_total": int(len(recon_res)),
            "n_significant_fdr_0p05_total": int(recon_res["significant_fdr_0p05"].sum()) if not recon_res.empty else 0,
            "subgroup_sizes_by_target": recon_subgroups,
            "feature_columns": recon_feature_cols,
            "n_participants_merged": int(part_recon_mwu["participant"].nunique()),
        },
    }

    os.makedirs(OUTDIR, exist_ok=True)
    out_part_raw = os.path.join(OUTDIR, "eads_mwu_all_participants_participant_level.csv")
    out_tests_raw = os.path.join(OUTDIR, "eads_mwu_all_participants_results.csv")
    out_part_recon = os.path.join(OUTDIR, "eads_mwu_all_participants_recon_nrmse_participant_level.csv")
    out_sess_recon = os.path.join(OUTDIR, "eads_recon_nrmse_all_sessions.csv")
    out_tests_recon = os.path.join(OUTDIR, "eads_mwu_all_participants_recon_nrmse_results.csv")
    out_recon_ref = os.path.join(OUTDIR, "eads_mwu_all_participants_recon_nrmse_reference_stats.csv")
    out_summary = os.path.join(OUTDIR, "eads_mwu_all_participants_summary.json")

    part_raw.to_csv(out_part_raw, index=False)
    raw_res.to_csv(out_tests_raw, index=False)
    part_recon_mwu.to_csv(out_part_recon, index=False)
    sess_recon.to_csv(out_sess_recon, index=False)
    recon_res.to_csv(out_tests_recon, index=False)
    recon_ref_df.to_csv(out_recon_ref, index=False)
    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Remove old profile-nRMSE outputs to avoid confusion.
    legacy_profile_outputs = [
        os.path.join(OUTDIR, "eads_mwu_all_participants_nrmse_participant_level.csv"),
        os.path.join(OUTDIR, "eads_mwu_all_participants_nrmse_results.csv"),
        os.path.join(OUTDIR, "eads_mwu_all_participants_nrmse_reference_stats.csv"),
    ]
    for path in legacy_profile_outputs:
        if os.path.exists(path):
            os.remove(path)

    print("Exploratory MWU (all participants) complete.")
    print(f" - merged participants: {summary['n_participants_merged']}")
    print(f" - raw-feature tests: {summary['raw_feature_analysis']['n_tests_total']}")
    print(f" - raw-feature significant after FDR: {summary['raw_feature_analysis']['n_significant_fdr_0p05_total']}")
    print(f" - reconstruction-nRMSE tests: {summary['reconstruction_nrmse_analysis']['n_tests_total']}")
    print(
        " - reconstruction-nRMSE significant after FDR: "
        f"{summary['reconstruction_nrmse_analysis']['n_significant_fdr_0p05_total']}"
    )
    print(
        "Saved outputs:\n"
        f" - {out_part_raw}\n"
        f" - {out_tests_raw}\n"
        f" - {out_part_recon}\n"
        f" - {out_sess_recon}\n"
        f" - {out_tests_recon}\n"
        f" - {out_recon_ref}\n"
        f" - {out_summary}"
    )


if __name__ == "__main__":
    main()
