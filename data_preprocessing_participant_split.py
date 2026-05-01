# data_preprocessing_participant_split.py
# Versão simplificada: parâmetros definidos no topo do ficheiro (sem argparse)

import json
import os
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------
# CONFIG — edita aqui se precisares
# -------------------------
INPUT_CSV = "action_sequences_lstm_features.csv"  # <<-- ajusta para o caminho correto
SESSION_COL = "session_key"
PARTICIPANT_COL = "participant_id"
OUTDIR = "cv_outputs"
TRAIN_PCT = 0.80
SEED = 42
STRATIFY_COL: Optional[str] = None  # ou o nome da coluna para estratificar, ex.: "n_sessions_bin"
CSV_SEP = ";"  # ajustar se o CSV usar vírgula
ENCODING = "utf-8"
# -------------------------


def extract_participant_id(session_key: str) -> str:
    """Extract participant id from session_key formatted as 'participant|session|module'."""
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


def _participant_strat_labels(
    df: pd.DataFrame, participant_col: str, stratify_col: str
) -> pd.Series:
    """Build one stratification label per participant (mode over participant rows)."""
    if stratify_col not in df.columns:
        raise ValueError(f"stratify_col '{stratify_col}' not found in DataFrame.")

    work = df[[participant_col, stratify_col]].dropna().copy()
    if work.empty:
        raise ValueError(f"No non-null values found for stratify_col '{stratify_col}'.")

    def mode_or_first(values: pd.Series):
        m = values.mode()
        return m.iloc[0] if not m.empty else values.iloc[0]

    labels = work.groupby(participant_col)[stratify_col].agg(mode_or_first)
    return labels


def create_train_test_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_pct: float = 0.80,
    stratify_col: Optional[str] = None,
    participant_col: str = "participant_id",
) -> Dict[str, List[str]]:
    """
    Create participant-level train/test split.

    Returns:
        {
          "participants_train": [...],
          "participants_test": [...]
        }
    """
    if participant_col not in df.columns:
        raise ValueError(f"participant_col '{participant_col}' not found in DataFrame.")
    if not (0.0 < train_pct < 1.0):
        raise ValueError("train_pct must be in (0, 1).")

    participants = (
        df[participant_col].dropna().astype(str).drop_duplicates().sort_values().tolist()
    )
    if len(participants) < 2:
        raise ValueError("Need at least two participants to create train/test split.")

    stratify_values = None
    if stratify_col:
        labels = _participant_strat_labels(df, participant_col=participant_col, stratify_col=stratify_col)
        label_map = labels.to_dict()
        missing = [p for p in participants if p not in label_map]
        if missing:
            sample = ", ".join(missing[:5])
            raise ValueError(
                f"Could not build stratify label for {len(missing)} participants (e.g., {sample})."
            )
        stratify_values = [label_map[p] for p in participants]

    participants_train, participants_test = train_test_split(
        participants,
        train_size=train_pct,
        random_state=seed,
        shuffle=True,
        stratify=stratify_values,
    )
    participants_train = sorted(map(str, participants_train))
    participants_test = sorted(map(str, participants_test))

    inter = set(participants_train).intersection(participants_test)
    if inter:
        raise RuntimeError(f"Leakage detected: train/test participant overlap: {sorted(inter)}")

    reconstructed = sorted(participants_train + participants_test)
    if reconstructed != sorted(participants):
        raise RuntimeError("Split coverage mismatch: train+test participants != all participants.")

    return {
        "participants_train": participants_train,
        "participants_test": participants_test,
    }


def _write_txt_list(path: str, values: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for v in values:
            fh.write(f"{v}\n")


def run_split():
    os.makedirs(OUTDIR, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, sep=CSV_SEP, encoding=ENCODING)
    if SESSION_COL not in df.columns:
        raise ValueError(f"session_col '{SESSION_COL}' not found in input CSV.")

    df = df.copy()
    df[PARTICIPANT_COL] = df[SESSION_COL].astype(str).apply(extract_participant_id)

    split = create_train_test_split(
        df=df,
        seed=SEED,
        train_pct=TRAIN_PCT,
        stratify_col=STRATIFY_COL,
        participant_col=PARTICIPANT_COL,
    )

    participants_train = split["participants_train"]
    participants_test = split["participants_test"]

    sessions_train = df[df[PARTICIPANT_COL].isin(participants_train)].copy()
    sessions_test = df[df[PARTICIPANT_COL].isin(participants_test)].copy()

    if sessions_train.empty or sessions_test.empty:
        raise RuntimeError(
            "Empty train/test rows after participant split. Check input data and split settings."
        )

    train_out = os.path.join(OUTDIR, "sessions_train.csv")
    test_out = os.path.join(OUTDIR, "sessions_test.csv")
    train_txt = os.path.join(OUTDIR, "participants_train.txt")
    test_txt = os.path.join(OUTDIR, "participants_test.txt")
    split_json = os.path.join(OUTDIR, "participants_split.json")

    sessions_train.to_csv(train_out, sep=CSV_SEP, index=False, encoding=ENCODING)
    sessions_test.to_csv(test_out, sep=CSV_SEP, index=False, encoding=ENCODING)
    _write_txt_list(train_txt, participants_train)
    _write_txt_list(test_txt, participants_test)

    payload = {
        "seed": SEED,
        "train_pct": TRAIN_PCT,
        "stratify_col": STRATIFY_COL,
        "participant_col": PARTICIPANT_COL,
        "session_col": SESSION_COL,
        "n_participants_total": int(df[PARTICIPANT_COL].nunique()),
        "n_participants_train": len(participants_train),
        "n_participants_test": len(participants_test),
        "n_sessions_total": int(df[SESSION_COL].astype(str).nunique()),
        "n_sessions_train": int(sessions_train[SESSION_COL].astype(str).nunique()),
        "n_sessions_test": int(sessions_test[SESSION_COL].astype(str).nunique()),
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(sessions_train)),
        "n_rows_test": int(len(sessions_test)),
        "participants_train": participants_train,
        "participants_test": participants_test,
    }
    with open(split_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("Saved participant split artifacts:")
    print(f" - {train_out}")
    print(f" - {test_out}")
    print(f" - {train_txt}")
    print(f" - {test_txt}")
    print(f" - {split_json}")
    print(
        f"Integrity check: overlap={len(set(participants_train).intersection(participants_test))} (must be 0)"
    )


if __name__ == "__main__":
    run_split()