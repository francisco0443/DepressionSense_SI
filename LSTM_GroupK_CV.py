# LSTM_GroupK_CV.py
# Versão simplificada: parâmetros definidos no topo do ficheiro (sem argparse)

import json
import os
import random
import time
import copy
from typing import Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Embedding,
    Input,
    LSTM,
    Lambda,
    Masking,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_preprocessing_LSTM import (
    handle_missing_values,
    qualitative_feature_preprocessing,
    scalefeatures,
    transform_features,
)

# -------------------------
# CONFIG — edita aqui se precisares
# -------------------------
INPUT_CSV = "action_sequences_lstm_features.csv"                 # caminho do CSV de ações (rows)
PARTICIPANTS_TRAIN_TXT = "cv_outputs/participants_train.txt"     # ficheiro com participantes (train)
PARTICIPANTS_TEST_TXT = "cv_outputs/participants_test.txt"       # ficheiro com participantes (test)
GROUP_COL = "participant_id"
SESSION_COL = "session_key"
INNER_SPLITS = 3
PARAM_GRID_JSON = None      # full grid run
SEED = 42
OUTDIR = "cv_outputs"
CSV_SEP = ";"
ENCODING = "utf-8"
# -------------------------

CATEGORICAL_FEATURES = [
    "module_id",
    "action_type",
    "dom_element_type",
    "has_duration",
    "has_kinematics",
]
DEFAULT_PARAM_GRID = {
    "lstm_units": [32, 64, 128],
    "bottleneck_dim": [4, 8],
    "learning_rate": [1e-2, 1e-3, 1e-4],
    "dropout_rate": [0.2],
    "batch_size": [32],
}
DROP_XY_MODEL = True
MAXLEN_PERCENTILE = 90
INNER_MAX_EPOCHS = 25
INNER_PATIENCE = 5
BATCH_SIZE_PRED = 32
LENGTH_BIN_LABELS = ["very_short", "short", "medium", "long", "very_long"]

_LAST_PREPARE_SEQUENCES_STATS: Optional[Dict] = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _compute_truncation_stats(
    seq_lengths: List[int],
    participant_ids: List[str],
    maxlen: int,
    percentile_for_maxlen: int,
) -> Dict:
    if len(seq_lengths) != len(participant_ids):
        raise ValueError("seq_lengths and participant_ids must have the same length.")

    n_sessions_total = int(len(seq_lengths))
    n_steps_total = int(np.sum(seq_lengths)) if seq_lengths else 0
    n_sessions_truncated = 0
    n_steps_removed_total = 0

    by_participant: Dict[str, Dict[str, int]] = {}
    for pid, length in zip(participant_ids, seq_lengths):
        pid = str(pid)
        length = int(length)
        if pid not in by_participant:
            by_participant[pid] = {"sessions_truncated": 0, "steps_removed": 0}

        removed = max(0, length - int(maxlen))
        if removed > 0:
            n_sessions_truncated += 1
            by_participant[pid]["sessions_truncated"] += 1
        n_steps_removed_total += int(removed)
        by_participant[pid]["steps_removed"] += int(removed)

    by_participant_sorted = {k: by_participant[k] for k in sorted(by_participant.keys())}
    participants_with_truncation = sorted(
        [k for k, v in by_participant_sorted.items() if int(v["sessions_truncated"]) > 0]
    )

    percent_sessions_truncated = (
        100.0 * float(n_sessions_truncated) / float(n_sessions_total)
        if n_sessions_total > 0
        else 0.0
    )
    percent_steps_removed = (
        100.0 * float(n_steps_removed_total) / float(n_steps_total)
        if n_steps_total > 0
        else 0.0
    )

    return {
        "maxlen_used": int(maxlen),
        "maxlen_percentile": int(percentile_for_maxlen),
        "n_sessions_total": int(n_sessions_total),
        "n_sessions_truncated": int(n_sessions_truncated),
        "percent_sessions_truncated": float(percent_sessions_truncated),
        "n_steps_total": int(n_steps_total),
        "n_steps_removed_total": int(n_steps_removed_total),
        "percent_steps_removed": float(percent_steps_removed),
        "participants_with_truncation": participants_with_truncation,
        "by_participant": by_participant_sorted,
    }


def _set_last_prepare_sequences_stats(stats: Dict) -> None:
    global _LAST_PREPARE_SEQUENCES_STATS
    _LAST_PREPARE_SEQUENCES_STATS = copy.deepcopy(stats)


def get_last_prepare_sequences_stats() -> Dict:
    if _LAST_PREPARE_SEQUENCES_STATS is None:
        raise RuntimeError("No prepare_sequences truncation stats are available.")
    return copy.deepcopy(_LAST_PREPARE_SEQUENCES_STATS)


def build_length_bin_edges_from_train(train_lengths: pd.Series) -> np.ndarray:
    if train_lengths.empty:
        raise ValueError("Cannot build length bins from empty train lengths.")
    q = train_lengths.quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(float).to_numpy()
    edges = q.copy()
    for i in range(1, len(edges) - 1):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def extract_participant_id(session_key: str) -> str:
    if not isinstance(session_key, str):
        raise ValueError(f"Invalid session_key type: {type(session_key)}. Expected str.")
    key = session_key.strip()
    if not key or "|" not in key:
        raise ValueError(
            f"Malformed session_key '{session_key}'. Expected format 'participant|session|module'."
        )
    participant_id = key.split("|", 1)[0].strip()
    if not participant_id:
        raise ValueError(
            f"Malformed session_key '{session_key}'. Participant prefix is empty."
        )
    return participant_id


def read_participants_txt(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Participant list not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        vals = [line.strip() for line in fh.readlines()]
    vals = [v for v in vals if v]
    if not vals:
        raise ValueError(f"Participant list is empty: {path}")
    return sorted(set(vals))


def write_participants_txt(path: str, participants: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for p in participants:
            fh.write(f"{p}\n")


def load_param_grid(path: Optional[str]) -> List[Dict]:
    required = {"lstm_units", "bottleneck_dim", "learning_rate", "dropout_rate", "batch_size"}

    if path is None:
        grid = list(ParameterGrid(DEFAULT_PARAM_GRID))
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"param_grid_json file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        # Accept `best_params.json` produced by this script.
        if isinstance(raw, dict) and "best_params" in raw:
            if not isinstance(raw["best_params"], dict):
                raise ValueError("'best_params' must be a JSON object.")
            grid = [raw["best_params"]]
        # Accept direct single-config dictionaries (scalar values).
        elif isinstance(raw, dict) and required.issubset(raw.keys()) and all(
            not isinstance(v, (list, tuple, set)) for v in raw.values()
        ):
            grid = [raw]
        # Accept list with a single direct config dict (scalar values).
        elif (
            isinstance(raw, list)
            and len(raw) == 1
            and isinstance(raw[0], dict)
            and required.issubset(raw[0].keys())
            and all(not isinstance(v, (list, tuple, set)) for v in raw[0].values())
        ):
            grid = [raw[0]]
        else:
            if not isinstance(raw, (dict, list)):
                raise ValueError("param_grid_json must contain a JSON object or list of objects.")
            grid = list(ParameterGrid(raw))

    if not grid:
        raise ValueError("Parameter grid is empty.")

    for i, params in enumerate(grid, start=1):
        miss = required.difference(params.keys())
        if miss:
            raise ValueError(f"Config {i} missing required keys: {sorted(miss)}")
    return grid


def ensure_group_column(df: pd.DataFrame, session_col: str, group_col: str) -> pd.DataFrame:
    out = df.copy()
    if group_col in out.columns:
        out[group_col] = out[group_col].astype(str)
        return out
    if group_col != "participant_id":
        raise ValueError(
            f"group_col '{group_col}' not found in input and automatic extraction is only supported for 'participant_id'."
        )
    if session_col not in out.columns:
        raise ValueError(f"session_col '{session_col}' not found in input data.")
    out[group_col] = out[session_col].astype(str).apply(extract_participant_id)
    return out


def preprocess_fit_transform(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Dict]:
    train_proc = handle_missing_values(train_df, save=False)
    val_proc = handle_missing_values(val_df, save=False)

    train_proc = transform_features(train_proc, save=False)
    val_proc = transform_features(val_proc, save=False)

    train_proc, cat_maps = qualitative_feature_preprocessing(
        train_proc, fit=True, rare_min_count=20, save=False
    )
    val_proc, _ = qualitative_feature_preprocessing(
        val_proc, mappings=cat_maps, fit=False, save=False
    )

    train_proc, scalers, clip_params = scalefeatures(
        train_proc, fit=True, drop_xy=DROP_XY_MODEL, save=False
    )
    val_proc, _, _ = scalefeatures(
        val_proc,
        fit=False,
        scalers=scalers,
        clip_params=clip_params,
        drop_xy=DROP_XY_MODEL,
        save=False,
    )
    return train_proc, val_proc, cat_maps, scalers, clip_params


def preprocess_fit_only(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    train_proc = handle_missing_values(df, save=False)
    train_proc = transform_features(train_proc, save=False)
    train_proc, cat_maps = qualitative_feature_preprocessing(
        train_proc, fit=True, rare_min_count=20, save=False
    )
    train_proc, scalers, clip_params = scalefeatures(
        train_proc, fit=True, drop_xy=DROP_XY_MODEL, save=False
    )
    return train_proc, cat_maps, scalers, clip_params


def preprocess_transform_only(
    df: pd.DataFrame, cat_maps: Dict, scalers: Dict, clip_params: Dict
) -> pd.DataFrame:
    out = handle_missing_values(df, save=False)
    out = transform_features(out, save=False)
    out, _ = qualitative_feature_preprocessing(out, mappings=cat_maps, fit=False, save=False)
    out, _, _ = scalefeatures(
        out,
        fit=False,
        scalers=scalers,
        clip_params=clip_params,
        drop_xy=DROP_XY_MODEL,
        save=False,
    )
    return out


def prepare_sequences(
    df_data: pd.DataFrame,
    session_col: str,
    group_col: str,
    categorical_features: List[str],
    percentile_for_maxlen: int = MAXLEN_PERCENTILE,
    maxlen: Optional[int] = None,
    continuous_features_ref: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, List[str], List[str], List[str]]:
    grouped = list(df_data.groupby(session_col, sort=True))
    if not grouped:
        raise ValueError("No sessions found to build sequences.")

    exclude_cols = [session_col, group_col, "x_raw", "y_raw"] + categorical_features
    if DROP_XY_MODEL:
        exclude_cols += ["x", "y"]

    if continuous_features_ref is None:
        continuous_features = [c for c in df_data.columns if c not in exclude_cols]
    else:
        missing = [c for c in continuous_features_ref if c not in df_data.columns]
        if missing:
            raise ValueError(f"Missing continuous features in data: {missing}")
        continuous_features = list(continuous_features_ref)

    cont_seqs_unpadded: List[np.ndarray] = []
    cat_seqs_unpadded: Dict[str, List[np.ndarray]] = {f: [] for f in categorical_features}
    session_ids: List[str] = []
    participant_ids: List[str] = []

    for sid, g in grouped:
        sid = str(sid)
        session_ids.append(sid)

        pids = g[group_col].astype(str).unique()
        if len(pids) != 1:
            raise ValueError(f"Session '{sid}' is linked to multiple participants: {pids.tolist()}")
        participant_ids.append(str(pids[0]))

        cont_arr = g[continuous_features].values.astype(np.float32)
        cont_seqs_unpadded.append(cont_arr)

        for f in categorical_features:
            if f not in g.columns:
                raise ValueError(f"Categorical feature '{f}' missing from processed DataFrame.")
            if g[f].isna().any():
                raise ValueError(f"Categorical feature '{f}' contains NaN after preprocessing.")
            cat_vals = g[f].astype(np.int32).values + 1
            cat_seqs_unpadded[f].append(cat_vals)

    seq_lengths = [len(s) for s in cont_seqs_unpadded]
    if maxlen is None:
        maxlen = int(np.percentile(seq_lengths, percentile_for_maxlen))
        if maxlen < 1:
            maxlen = max(seq_lengths) if seq_lengths else 1

    truncation_stats = _compute_truncation_stats(
        seq_lengths=seq_lengths,
        participant_ids=participant_ids,
        maxlen=int(maxlen),
        percentile_for_maxlen=int(percentile_for_maxlen),
    )
    _set_last_prepare_sequences_stats(truncation_stats)

    cont_padded = pad_sequences(
        cont_seqs_unpadded,
        maxlen=maxlen,
        dtype="float32",
        value=0.0,
        padding="post",
        truncating="post",
    )
    cat_padded: Dict[str, np.ndarray] = {}
    for f in categorical_features:
        cat_padded[f] = pad_sequences(
            cat_seqs_unpadded[f],
            maxlen=maxlen,
            dtype="int32",
            value=0,
            padding="post",
            truncating="post",
        )

    return cont_padded, cat_padded, int(maxlen), continuous_features, session_ids, participant_ids


def build_embedding_config(
    cat_train: Dict[str, np.ndarray], categorical_features: List[str]
) -> Dict[str, Tuple[int, int]]:
    cfg = {}
    for f in categorical_features:
        max_id = int(np.max(cat_train[f]))
        input_dim = max_id + 1
        cardinality = max(0, input_dim - 1)
        emb_dim = min(50, max(1, (cardinality // 2) + 1))
        cfg[f] = (input_dim, emb_dim)
    return cfg


def build_lstm_autoencoder(
    timesteps: int,
    num_cont_features: int,
    categorical_features_list: List[str],
    embedding_cfg: Dict[str, Tuple[int, int]],
    lstm_units: int,
    bottleneck_dim: int,
    dropout_rate: float,
) -> Model:
    cont_input = Input(shape=(timesteps, num_cont_features), name="continuous_input")

    cat_inputs, cat_embeds = [], []
    for f in categorical_features_list:
        inp = Input(shape=(timesteps,), dtype="int32", name=f)
        if f not in embedding_cfg:
            raise ValueError(f"Missing embedding config for categorical '{f}'.")
        input_dim, output_dim = embedding_cfg[f]

        emb_raw = Embedding(
            input_dim=input_dim,
            output_dim=output_dim,
            mask_zero=False,
            name=f"{f}_embedding",
        )(inp)
        pad_mask = Lambda(
            lambda z: tf.cast(tf.not_equal(z, 0), tf.float32)[..., None],
            name=f"{f}_padmask",
        )(inp)
        emb = Lambda(lambda args: args[0] * args[1], name=f"{f}_masked_emb")([emb_raw, pad_mask])

        cat_inputs.append(inp)
        cat_embeds.append(emb)

    concat = Concatenate(axis=-1, name="concat_inputs")([cont_input] + cat_embeds)
    masked = Masking(mask_value=0.0, name="mask_after_concat")(concat)
    encoded = LSTM(
        lstm_units, activation="tanh", dropout=dropout_rate, name="encoder_lstm"
    )(masked)
    bottleneck = Dense(
        bottleneck_dim,
        activation="tanh",
        name="bottleneck",
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),
    )(encoded)
    repeated = RepeatVector(timesteps, name="repeat")(bottleneck)
    decoded = LSTM(
        lstm_units,
        activation="tanh",
        return_sequences=True,
        dropout=dropout_rate,
        name="decoder_lstm",
    )(repeated)
    cont_out = TimeDistributed(Dense(num_cont_features), name="reconstructed_continuous")(decoded)

    return Model(inputs=[cont_input] + cat_inputs, outputs=cont_out)


def make_mask(cat_dict: Dict[str, np.ndarray], action_type_name: str = "action_type") -> np.ndarray:
    return (cat_dict[action_type_name] != 0)


def per_session_mse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_f = mask[..., None].astype(np.float64)
    n_features = y_true.shape[2]
    se_i = (((y_true - y_pred) ** 2) * mask_f).sum(axis=(1, 2))
    cnt_i = mask.sum(axis=1).astype(np.float64)
    cnt_i[cnt_i == 0] = 1.0
    return se_i / (cnt_i * n_features)


def feature_level_reconstruction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute feature-level reconstruction metrics on valid (non-padding) timesteps only.
    nRMSE is provided with multiple normalizations; nRMSE_by_std is used as primary.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.ndim != 3:
        raise ValueError("Expected y_true/y_pred shape (n_sessions, timesteps, n_features).")
    if mask.shape != y_true.shape[:2]:
        raise ValueError("mask must match first two dimensions of y_true.")
    if len(feature_names) != y_true.shape[2]:
        raise ValueError("feature_names length must match number of features.")

    valid = mask.astype(bool)
    rows: List[Dict] = []

    for j, feat in enumerate(feature_names):
        yt = y_true[:, :, j][valid].astype(np.float64)
        yp = y_pred[:, :, j][valid].astype(np.float64)
        n_valid = int(yt.size)
        if n_valid == 0:
            rows.append(
                {
                    "feature": feat,
                    "n_valid_points": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mse": np.nan,
                    "mean_true": np.nan,
                    "std_true": np.nan,
                    "range_true": np.nan,
                    "iqr_true": np.nan,
                    "nrmse_by_std": np.nan,
                    "nrmse_by_range": np.nan,
                    "nrmse_by_iqr": np.nan,
                }
            )
            continue

        err = yt - yp
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
        mean_true = float(np.mean(yt))
        std_true = float(np.std(yt))
        range_true = float(np.max(yt) - np.min(yt))
        q25, q75 = np.percentile(yt, [25, 75])
        iqr_true = float(q75 - q25)

        nrmse_std = float(rmse / std_true) if std_true > 0 else np.nan
        nrmse_range = float(rmse / range_true) if range_true > 0 else np.nan
        nrmse_iqr = float(rmse / iqr_true) if iqr_true > 0 else np.nan

        rows.append(
            {
                "feature": feat,
                "n_valid_points": n_valid,
                "mae": mae,
                "rmse": rmse,
                "mse": mse,
                "mean_true": mean_true,
                "std_true": std_true,
                "range_true": range_true,
                "iqr_true": iqr_true,
                "nrmse_by_std": nrmse_std,
                "nrmse_by_range": nrmse_range,
                "nrmse_by_iqr": nrmse_iqr,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(by="nrmse_by_std", ascending=False, na_position="last").reset_index(
        drop=True
    )
    out["rank_nrmse_by_std"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def build_inner_splits(
    train_df: pd.DataFrame,
    session_col: str,
    group_col: str,
    inner_splits: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    session_table = train_df[[session_col, group_col]].drop_duplicates().copy()
    session_table[session_col] = session_table[session_col].astype(str)
    session_table[group_col] = session_table[group_col].astype(str)
    session_table = session_table.sort_values(session_col).reset_index(drop=True)

    gkf = GroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    folds: List[Tuple[List[str], List[str]]] = []

    for tr_idx, va_idx in gkf.split(session_table, groups=session_table[group_col]):
        train_sids = session_table.iloc[tr_idx][session_col].astype(str).tolist()
        val_sids = session_table.iloc[va_idx][session_col].astype(str).tolist()

        train_groups = set(session_table.iloc[tr_idx][group_col].astype(str).tolist())
        val_groups = set(session_table.iloc[va_idx][group_col].astype(str).tolist())
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise RuntimeError(f"Group leakage inside inner CV fold: {sorted(overlap)}")

        folds.append((train_sids, val_sids))
    return folds


def main():
    start_time = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    set_global_seed(SEED)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"input_csv not found: {INPUT_CSV}")

    base_df = pd.read_csv(INPUT_CSV, sep=CSV_SEP, encoding=ENCODING)
    if SESSION_COL not in base_df.columns:
        raise ValueError(f"session_col '{SESSION_COL}' not found in input CSV.")
    base_df = ensure_group_column(base_df, session_col=SESSION_COL, group_col=GROUP_COL)

    participants_train = read_participants_txt(PARTICIPANTS_TRAIN_TXT)
    participants_test = read_participants_txt(PARTICIPANTS_TEST_TXT)
    overlap = set(participants_train).intersection(participants_test)
    if overlap:
        raise RuntimeError(f"participants_train ∩ participants_test is not empty: {sorted(overlap)}")

    observed_groups = set(base_df[GROUP_COL].astype(str).unique())
    missing_train = sorted(set(participants_train).difference(observed_groups))
    missing_test = sorted(set(participants_test).difference(observed_groups))
    if missing_train:
        raise ValueError(f"{len(missing_train)} train participants not found in data (e.g. {missing_train[:5]})")
    if missing_test:
        raise ValueError(f"{len(missing_test)} test participants not found in data (e.g. {missing_test[:5]})")

    train_df = base_df[base_df[GROUP_COL].astype(str).isin(participants_train)].copy()
    test_df = base_df[base_df[GROUP_COL].astype(str).isin(participants_test)].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError("train_df/test_df is empty after applying participant lists.")

    # Persist split artifacts in outdir
    sessions_train_csv = os.path.join(OUTDIR, "sessions_train.csv")
    sessions_test_csv = os.path.join(OUTDIR, "sessions_test.csv")
    participants_train_out = os.path.join(OUTDIR, "participants_train.txt")
    participants_test_out = os.path.join(OUTDIR, "participants_test.txt")
    participants_split_json = os.path.join(OUTDIR, "participants_split.json")

    train_df.to_csv(sessions_train_csv, sep=CSV_SEP, index=False, encoding=ENCODING)
    test_df.to_csv(sessions_test_csv, sep=CSV_SEP, index=False, encoding=ENCODING)
    write_participants_txt(participants_train_out, participants_train)
    write_participants_txt(participants_test_out, participants_test)
    with open(participants_split_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "seed": SEED,
                "group_col": GROUP_COL,
                "session_col": SESSION_COL,
                "n_participants_train": len(participants_train),
                "n_participants_test": len(participants_test),
                "n_sessions_train": int(train_df[SESSION_COL].astype(str).nunique()),
                "n_sessions_test": int(test_df[SESSION_COL].astype(str).nunique()),
                "n_rows_train": int(len(train_df)),
                "n_rows_test": int(len(test_df)),
                "participants_train": participants_train,
                "participants_test": participants_test,
            },
            fh,
            indent=2,
        )

    folds = build_inner_splits(
        train_df=train_df,
        session_col=SESSION_COL,
        group_col=GROUP_COL,
        inner_splits=INNER_SPLITS,
        seed=SEED,
    )
    params_list = load_param_grid(PARAM_GRID_JSON)
    print(f"Running inner GroupKFold with {INNER_SPLITS} folds and {len(params_list)} configs.")

    inner_rows: List[Dict] = []
    truncation_inner_rows: List[Dict] = []
    train_sessions_all = set(train_df[SESSION_COL].astype(str).unique())

    for cfg_idx, params in enumerate(params_list, start=1):
        fold_losses: List[float] = []
        fold_best_epochs: List[float] = []
        print(f"\nConfig {cfg_idx}/{len(params_list)}: {params}")

        for fold_idx, (fold_train_sids, fold_val_sids) in enumerate(folds, start=1):
            fold_train_set = set(fold_train_sids)
            fold_val_set = set(fold_val_sids)

            if fold_train_set.intersection(fold_val_set):
                raise RuntimeError(f"Session overlap in fold {fold_idx} for config {cfg_idx}.")
            if fold_train_set.union(fold_val_set) != train_sessions_all:
                raise RuntimeError(
                    f"Fold coverage mismatch in fold {fold_idx}, config {cfg_idx}."
                )

            fold_train_df = train_df[train_df[SESSION_COL].astype(str).isin(fold_train_set)].copy()
            fold_val_df = train_df[train_df[SESSION_COL].astype(str).isin(fold_val_set)].copy()
            if fold_train_df.empty or fold_val_df.empty:
                raise RuntimeError(
                    f"Empty fold train/val at fold {fold_idx}, config {cfg_idx}."
                )

            fold_seed = SEED + cfg_idx * 1000 + fold_idx
            set_global_seed(fold_seed)

            try:
                tr_proc, va_proc, _, _, _ = preprocess_fit_transform(fold_train_df, fold_val_df)
                X_tr_cont, X_tr_cat, fold_maxlen, cont_features, _, _ = prepare_sequences(
                    tr_proc,
                    session_col=SESSION_COL,
                    group_col=GROUP_COL,
                    categorical_features=CATEGORICAL_FEATURES,
                    percentile_for_maxlen=MAXLEN_PERCENTILE,
                    maxlen=None,
                )
                fold_trunc_train = get_last_prepare_sequences_stats()
                X_va_cont, X_va_cat, _, _, _, _ = prepare_sequences(
                    va_proc,
                    session_col=SESSION_COL,
                    group_col=GROUP_COL,
                    categorical_features=CATEGORICAL_FEATURES,
                    maxlen=fold_maxlen,
                    continuous_features_ref=cont_features,
                )
                fold_trunc_val = get_last_prepare_sequences_stats()
                truncation_inner_rows.append(
                    {
                        "config_id": int(cfg_idx),
                        "fold_id": int(fold_idx),
                        "train": fold_trunc_train,
                        "val": fold_trunc_val,
                    }
                )

                embedding_cfg = build_embedding_config(X_tr_cat, CATEGORICAL_FEATURES)
                model = build_lstm_autoencoder(
                    timesteps=fold_maxlen,
                    num_cont_features=X_tr_cont.shape[2],
                    categorical_features_list=CATEGORICAL_FEATURES,
                    embedding_cfg=embedding_cfg,
                    lstm_units=int(params["lstm_units"]),
                    bottleneck_dim=int(params["bottleneck_dim"]),
                    dropout_rate=float(params["dropout_rate"]),
                )
                model.compile(
                    optimizer=Adam(learning_rate=float(params["learning_rate"])),
                    loss="mse",
                )

                X_tr_cat_list = [X_tr_cat[f] for f in CATEGORICAL_FEATURES]
                X_va_cat_list = [X_va_cat[f] for f in CATEGORICAL_FEATURES]
                w_tr = make_mask(X_tr_cat, "action_type").astype("float32")
                w_va = make_mask(X_va_cat, "action_type").astype("float32")

                callbacks = [
                    EarlyStopping(
                        monitor="val_loss",
                        patience=INNER_PATIENCE,
                        restore_best_weights=True,
                        verbose=0,
                    )
                ]
                history = model.fit(
                    [X_tr_cont] + X_tr_cat_list,
                    X_tr_cont,
                    sample_weight=w_tr,
                    validation_data=([X_va_cont] + X_va_cat_list, X_va_cont, w_va),
                    epochs=INNER_MAX_EPOCHS,
                    batch_size=int(params["batch_size"]),
                    callbacks=callbacks,
                    verbose=0,
                )

                if "val_loss" not in history.history or not history.history["val_loss"]:
                    fold_val_loss = float("inf")
                    best_epoch = np.nan
                else:
                    val_losses = np.array(history.history["val_loss"], dtype=np.float64)
                    best_idx = int(np.nanargmin(val_losses))
                    fold_val_loss = float(val_losses[best_idx])
                    best_epoch = float(best_idx + 1)

                fold_losses.append(fold_val_loss)
                fold_best_epochs.append(best_epoch)
                print(
                    f"  Fold {fold_idx}/{INNER_SPLITS} val_loss={fold_val_loss:.6f}, best_epoch={best_epoch}"
                )
            except Exception as exc:
                fold_losses.append(float("inf"))
                fold_best_epochs.append(np.nan)
                print(f"  Fold {fold_idx}/{INNER_SPLITS} failed: {exc}")
            finally:
                tf.keras.backend.clear_session()

        valid_epochs = [e for e in fold_best_epochs if np.isfinite(e)]
        median_best_epoch = float(np.median(valid_epochs)) if valid_epochs else float("nan")

        row = {
            "config_id": int(cfg_idx),
            "params_json": json.dumps(params, sort_keys=True),
            "mean_val_loss": float(np.mean(fold_losses)),
            "std_val_loss": float(np.std(fold_losses)),
            "median_best_epoch": median_best_epoch,
        }
        for i, loss in enumerate(fold_losses, start=1):
            row[f"fold_{i}_val_loss"] = float(loss)
        inner_rows.append(row)

    inner_df = pd.DataFrame(inner_rows)
    inner_results_csv = os.path.join(OUTDIR, "inner_search_results.csv")
    inner_df.to_csv(inner_results_csv, index=False)

    ranked = inner_df.sort_values(
        by=["mean_val_loss", "std_val_loss", "config_id"], ascending=[True, True, True]
    ).reset_index(drop=True)
    best_row = ranked.iloc[0]
    best_mean = float(best_row["mean_val_loss"])
    if not np.isfinite(best_mean):
        raise RuntimeError("All configurations failed during inner CV.")

    best_params = json.loads(best_row["params_json"])
    median_best_epoch = float(best_row["median_best_epoch"])
    if np.isfinite(median_best_epoch):
        epochs_final = max(1, int(round(median_best_epoch)))
    else:
        epochs_final = INNER_MAX_EPOCHS

    best_params_payload = {
        "best_params": best_params,
        "median_best_epoch": median_best_epoch,
        "epochs_final": int(epochs_final),
        "selection_rule": "min(mean_val_loss), tie-break by min(std_val_loss), then min(config_id)",
    }
    best_params_json = os.path.join(OUTDIR, "best_params.json")
    with open(best_params_json, "w", encoding="utf-8") as fh:
        json.dump(best_params_payload, fh, indent=2)

    set_global_seed(SEED)
    train_proc, cat_maps, scalers, clip_params = preprocess_fit_only(train_df)
    test_proc = preprocess_transform_only(test_df, cat_maps=cat_maps, scalers=scalers, clip_params=clip_params)

    X_tr_cont, X_tr_cat, timesteps, continuous_features, train_session_ids, train_participant_ids = prepare_sequences(
        train_proc,
        session_col=SESSION_COL,
        group_col=GROUP_COL,
        categorical_features=CATEGORICAL_FEATURES,
        percentile_for_maxlen=MAXLEN_PERCENTILE,
        maxlen=None,
    )
    final_train_truncation = get_last_prepare_sequences_stats()
    X_te_cont, X_te_cat, _, _, test_session_ids, test_participant_ids = prepare_sequences(
        test_proc,
        session_col=SESSION_COL,
        group_col=GROUP_COL,
        categorical_features=CATEGORICAL_FEATURES,
        maxlen=timesteps,
        continuous_features_ref=continuous_features,
    )
    final_test_truncation = get_last_prepare_sequences_stats()

    expected_test_sessions = sorted(test_df[SESSION_COL].astype(str).unique().tolist())
    got_test_sessions = sorted(test_session_ids)
    if expected_test_sessions != got_test_sessions:
        raise RuntimeError(
            "Test coverage mismatch: prepared test sequences do not match sessions_test."
        )

    embedding_cfg = build_embedding_config(X_tr_cat, CATEGORICAL_FEATURES)
    X_tr_cat_list = [X_tr_cat[f] for f in CATEGORICAL_FEATURES]
    X_te_cat_list = [X_te_cat[f] for f in CATEGORICAL_FEATURES]
    w_tr_bool = make_mask(X_tr_cat, "action_type")
    w_te_bool = make_mask(X_te_cat, "action_type")
    w_tr = w_tr_bool.astype("float32")

    set_global_seed(SEED)
    final_model = build_lstm_autoencoder(
        timesteps=timesteps,
        num_cont_features=X_tr_cont.shape[2],
        categorical_features_list=CATEGORICAL_FEATURES,
        embedding_cfg=embedding_cfg,
        lstm_units=int(best_params["lstm_units"]),
        bottleneck_dim=int(best_params["bottleneck_dim"]),
        dropout_rate=float(best_params["dropout_rate"]),
    )
    final_model.compile(
        optimizer=Adam(learning_rate=float(best_params["learning_rate"])),
        loss="mse",
    )

    history = final_model.fit(
        [X_tr_cont] + X_tr_cat_list,
        X_tr_cont,
        sample_weight=w_tr,
        epochs=int(epochs_final),
        batch_size=int(best_params["batch_size"]),
        verbose=2,
    )

    pred_te = final_model.predict(
        [X_te_cont] + X_te_cat_list, batch_size=BATCH_SIZE_PRED, verbose=0
    )

    w_tr_f = w_tr_bool[..., None].astype(np.float64)
    n_valid_train = float(w_tr_bool.sum()) if w_tr_bool.sum() > 0 else 1.0
    mean_f = (X_tr_cont * w_tr_f).sum(axis=(0, 1)) / n_valid_train
    baseline_pred_te = mean_f.reshape((1, 1, X_te_cont.shape[2]))

    mse_model = per_session_mse(X_te_cont, pred_te, w_te_bool)
    mse_baseline = per_session_mse(X_te_cont, baseline_pred_te, w_te_bool)
    improvement_pct = 100.0 * (mse_baseline - mse_model) / (mse_baseline + 1e-12)
    lengths_effective = w_te_bool.sum(axis=1).astype(int)

    train_lengths_real = train_proc.groupby(SESSION_COL).size().astype(int)
    test_lengths_real = test_proc.groupby(SESSION_COL).size().astype(int)
    missing_test_lengths = [sid for sid in test_session_ids if sid not in test_lengths_real.index]
    if missing_test_lengths:
        raise RuntimeError(
            f"Missing raw test lengths for {len(missing_test_lengths)} sessions."
        )
    lengths_raw = np.array([int(test_lengths_real.loc[sid]) for sid in test_session_ids], dtype=int)

    test_metrics_df = pd.DataFrame(
        {
            "session_id": test_session_ids,
            "participant_id": test_participant_ids,
            "mse_model": mse_model,
            "mse_baseline": mse_baseline,
            "improvement_pct": improvement_pct,
            "length": lengths_effective,
            "length_raw": lengths_raw,
        }
    )
    if test_metrics_df["session_id"].nunique() != len(expected_test_sessions):
        raise RuntimeError("test_session_metrics coverage failure: duplicate/missing session rows.")

    test_metrics_csv = os.path.join(OUTDIR, "test_session_metrics.csv")
    test_metrics_df.to_csv(test_metrics_csv, index=False)

    length_bin_edges = build_length_bin_edges_from_train(train_lengths_real)
    test_metrics_df["length_bin"] = pd.cut(
        test_metrics_df["length_raw"],
        bins=length_bin_edges,
        labels=LENGTH_BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    test_metrics_by_length_bin_df = (
        test_metrics_df.groupby("length_bin", observed=False)
        .agg(
            mean_mse_model=("mse_model", "mean"),
            std_mse_model=("mse_model", "std"),
            mean_improvement_pct=("improvement_pct", "mean"),
            count=("session_id", "count"),
        )
        .reindex(LENGTH_BIN_LABELS)
        .reset_index()
    )
    test_metrics_by_length_bin_df["length_bin"] = test_metrics_by_length_bin_df["length_bin"].astype(
        str
    )
    test_metrics_by_length_bin_csv = os.path.join(OUTDIR, "test_metrics_by_length_bin.csv")
    test_metrics_by_length_bin_df.to_csv(test_metrics_by_length_bin_csv, index=False)

    feature_metrics_df = feature_level_reconstruction_metrics(
        y_true=X_te_cont,
        y_pred=pred_te,
        mask=w_te_bool,
        feature_names=continuous_features,
    )
    feature_metrics_csv = os.path.join(OUTDIR, "feature_reconstruction_metrics.csv")
    feature_metrics_df.to_csv(feature_metrics_csv, index=False)

    top5_feature_metrics_df = feature_metrics_df.head(5).copy()
    top5_feature_metrics_csv = os.path.join(OUTDIR, "feature_reconstruction_top5_nrmse.csv")
    top5_feature_metrics_df.to_csv(top5_feature_metrics_csv, index=False)

    truncation_by_fold_payload = {
        "maxlen_percentile": int(MAXLEN_PERCENTILE),
        "padding": "post",
        "truncating": "post",
        "inner_cv": truncation_inner_rows,
        "final_train": final_train_truncation,
        "final_test": final_test_truncation,
    }
    truncation_by_fold_path = os.path.join(OUTDIR, "truncation_by_fold.json")
    with open(truncation_by_fold_path, "w", encoding="utf-8") as fh:
        json.dump(truncation_by_fold_payload, fh, indent=2)

    encoder_model = Model(inputs=final_model.inputs, outputs=final_model.get_layer("bottleneck").output)
    emb_tr = encoder_model.predict(
        [X_tr_cont] + X_tr_cat_list, batch_size=BATCH_SIZE_PRED, verbose=0
    )
    emb_te = encoder_model.predict(
        [X_te_cont] + X_te_cat_list, batch_size=BATCH_SIZE_PRED, verbose=0
    )

    if not np.isfinite(emb_te).all():
        raise RuntimeError("test embeddings contain NaN/Inf.")

    emb_raw_df = pd.DataFrame(
        {"session_id": test_session_ids, "participant_id": test_participant_ids}
    )
    for j in range(emb_te.shape[1]):
        emb_raw_df[f"emb_{j}"] = emb_te[:, j]
    test_emb_raw_csv = os.path.join(OUTDIR, "test_embeddings_raw.csv")
    emb_raw_df.to_csv(test_emb_raw_csv, index=False)

    emb_scaler = StandardScaler().fit(emb_tr)
    emb_te_std = emb_scaler.transform(emb_te)
    if not np.isfinite(emb_te_std).all():
        raise RuntimeError("standardized test embeddings contain NaN/Inf.")

    emb_std_df = pd.DataFrame(
        {"session_id": test_session_ids, "participant_id": test_participant_ids}
    )
    for j in range(emb_te_std.shape[1]):
        emb_std_df[f"emb_{j}"] = emb_te_std[:, j]
    test_emb_std_csv = os.path.join(OUTDIR, "test_embeddings_std.csv")
    emb_std_df.to_csv(test_emb_std_csv, index=False)

    weights_path = os.path.join(OUTDIR, "best_model_final.weights.h5")
    final_model.save_weights(weights_path)

    preprocessing_artifacts = {
        "cat_maps": cat_maps,
        "scalers": scalers,
        "clip_params": clip_params,
        "drop_xy_model": DROP_XY_MODEL,
        "maxlen_percentile": MAXLEN_PERCENTILE,
        "group_col": GROUP_COL,
        "session_col": SESSION_COL,
        "continuous_features": continuous_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "embedding_std_scaler": emb_scaler,
    }
    preprocessing_artifacts_path = os.path.join(OUTDIR, "train_preprocessing_artifacts.joblib")
    joblib.dump(preprocessing_artifacts, preprocessing_artifacts_path)

    metadata = {
        "timesteps": int(timesteps),
        "continuous_features": continuous_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "embedding_config": {k: [int(v[0]), int(v[1])] for k, v in embedding_cfg.items()},
        "best_params": {
            "lstm_units": int(best_params["lstm_units"]),
            "bottleneck_dim": int(best_params["bottleneck_dim"]),
            "learning_rate": float(best_params["learning_rate"]),
            "dropout_rate": float(best_params["dropout_rate"]),
            "batch_size": int(best_params["batch_size"]),
            "epochs_final": int(epochs_final),
        },
        "drop_xy_model": bool(DROP_XY_MODEL),
        "seed": int(SEED),
        "session_col": SESSION_COL,
        "group_col": GROUP_COL,
        "inner_splits": int(INNER_SPLITS),
        "maxlen_percentile_train_only": int(MAXLEN_PERCENTILE),
        "length_bins_from_train": {
            "labels": LENGTH_BIN_LABELS,
            "edges": [
                "-inf" if np.isneginf(x) else "inf" if np.isposinf(x) else float(x)
                for x in length_bin_edges.tolist()
            ],
        },
        "preprocessing_artifacts_path": os.path.basename(preprocessing_artifacts_path),
    }
    metadata_path = os.path.join(OUTDIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    runtime_seconds = float(time.time() - start_time)
    run_summary = {
        "seed": int(SEED),
        "input_csv": INPUT_CSV,
        "group_col": GROUP_COL,
        "session_col": SESSION_COL,
        "inner_splits": int(INNER_SPLITS),
        "maxlen_percentile": int(MAXLEN_PERCENTILE),
        "maxlen_used": int(final_train_truncation["maxlen_used"]),
        "percent_sessions_truncated": float(final_train_truncation["percent_sessions_truncated"]),
        "percent_steps_removed": float(final_train_truncation["percent_steps_removed"]),
        "participants_with_truncation": final_train_truncation["participants_with_truncation"],
        "n_param_configs": int(len(params_list)),
        "n_participants_train": int(len(participants_train)),
        "n_participants_test": int(len(participants_test)),
        "n_sessions_train": int(len(train_session_ids)),
        "n_sessions_test": int(len(test_session_ids)),
        "n_rows_train": int(len(train_df)),
        "n_rows_test": int(len(test_df)),
        "best_config_id": int(best_row["config_id"]),
        "best_mean_val_loss": float(best_row["mean_val_loss"]),
        "best_std_val_loss": float(best_row["std_val_loss"]),
        "epochs_final": int(epochs_final),
        "test_mean_mse_model": float(np.mean(mse_model)),
        "test_mean_mse_baseline": float(np.mean(mse_baseline)),
        "test_mean_improvement_pct": float(np.mean(improvement_pct)),
        "final_train_loss_last_epoch": float(history.history["loss"][-1]) if history.history.get("loss") else None,
        "truncation_by_fold_path": os.path.basename(truncation_by_fold_path),
        "test_metrics_by_length_bin_path": os.path.basename(test_metrics_by_length_bin_csv),
        "feature_reconstruction_metrics_path": os.path.basename(feature_metrics_csv),
        "feature_reconstruction_top5_nrmse_path": os.path.basename(top5_feature_metrics_csv),
        "runtime_seconds": runtime_seconds,
    }
    run_summary_path = os.path.join(OUTDIR, "run_summary.json")
    with open(run_summary_path, "w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2)

    print("\nSaved outputs:")
    print(f" - {sessions_train_csv}")
    print(f" - {sessions_test_csv}")
    print(f" - {participants_train_out}")
    print(f" - {participants_test_out}")
    print(f" - {participants_split_json}")
    print(f" - {inner_results_csv}")
    print(f" - {best_params_json}")
    print(f" - {weights_path}")
    print(f" - {metadata_path}")
    print(f" - {preprocessing_artifacts_path}")
    print(f" - {test_metrics_csv}")
    print(f" - {test_metrics_by_length_bin_csv}")
    print(f" - {feature_metrics_csv}")
    print(f" - {top5_feature_metrics_csv}")
    print(f" - {test_emb_raw_csv}")
    print(f" - {test_emb_std_csv}")
    print(f" - {truncation_by_fold_path}")
    print(f" - {run_summary_path}")
    print("\nIntegrity checks passed:")
    print(f" - participants overlap: {len(overlap)}")
    print(f" - test sessions covered: {len(test_session_ids)}")


if __name__ == "__main__":
    main()
