import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl  # noqa: F401
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import warnings

# -----------------------------
# Data quality report
# -----------------------------
def assess_data_quality(df: pd.DataFrame, out_path: str = "data_summary_report.xlsx") -> None:
    print("\n=== Basic Info ===")
    print(df.info())
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    numeric_stats = df.describe().transpose()
    categorical_columns = ['module_id', 'action_type', 'dom_element_type', 'session_key']
    categorical_freqs = {col: df[col].value_counts() for col in categorical_columns if col in df.columns}
    print("\n=== Duplicates ===")
    print("Total duplicates:", df.duplicated().sum())
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="Reds")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.show()
    with pd.ExcelWriter(out_path) as writer:
        numeric_stats.to_excel(writer, sheet_name="Descriptive_Stats")
        for col, freq_df in categorical_freqs.items():
            freq_df.to_frame(name='frequency').to_excel(writer, sheet_name=f"{col}_freqs")
    print(f"Summary saved to '{out_path}'")

# -----------------------------
# Missing value handling
# -----------------------------
def handle_missing_values(df: pd.DataFrame, save: bool = False, out_path: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    kinematic_features = [
        'cursor_speed', 'acceleration_mean', 'reciprocal_acceleration',
        'jitter', 'direction_changes', 'curvature_mean', 'rate_of_curvature',
        'distance_travelled', 'movement_offset', 'straightness', 'self_intersections'
    ]
    present_cols = [c for c in kinematic_features if c in df.columns]
    if present_cols:
        df['has_kinematics'] = df[present_cols].notnull().all(axis=1).astype(int)
        df[present_cols] = df[present_cols].fillna(0)
    if 'action_duration' in df.columns:
        df['has_duration'] = df['action_duration'].notnull().astype(int)
        df['action_duration'] = df['action_duration'].fillna(0)
    if 'dom_element_type' in df.columns:
        df['dom_element_type'] = df['dom_element_type'].fillna('unknown')
    if save and out_path:
        df.to_csv(out_path, sep=";", index=False, encoding='utf-8')
        print(f"Saved updated CSV to {out_path}")
    return df

# -----------------------------
# Log1p and feature drops
# -----------------------------
def transform_features(df: pd.DataFrame, save: bool = False, out_path: Optional[str] = None) -> pd.DataFrame:
    log_transform_features = [
        'jitter', 'direction_changes', 'distance_travelled', 'movement_offset',
        'action_duration', 'time_since_last_action', 'cursor_speed', 'rate_of_curvature'
    ]
    drop_features = ['reciprocal_acceleration']
    df_transformed = df.copy()
    for feature in log_transform_features:
        if feature in df_transformed.columns:
            if (df_transformed[feature] < 0).any():
                warnings.warn(f"Feature {feature} contains negative values — log1p may be inappropriate.")
            df_transformed[feature] = np.log1p(df_transformed[feature].clip(lower=0))
    df_transformed = df_transformed.drop(columns=[f for f in drop_features if f in df_transformed.columns])
    if save and out_path:
        df_transformed.to_csv(out_path, sep=";", index=False, encoding='utf-8')
        print(f"Saved updated CSV to {out_path}")
    return df_transformed

# -----------------------------
# Categorical preprocessing
# -----------------------------
def qualitative_feature_preprocessing(df: pd.DataFrame, mappings: Optional[Dict] = None, rare_min_count: int = 20, fit: bool = False, save: bool = False, out_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    # Fixed mappings
    module_mapping = {'behavior_RESS': 0,'behavior_EADS': 1,'behavior_WAI_sr': 2}
    action_mapping = {'mouse_movement': 0,'hover': 1,'pause': 2,'click': 3}
    if 'module_id' in df.columns: df['module_id'] = df['module_id'].map(module_mapping)
    if 'action_type' in df.columns: df['action_type'] = df['action_type'].map(action_mapping)
    learned = mappings.copy() if mappings else {}
    if fit and 'dom_element_type' in df.columns:
        dom_counts = df['dom_element_type'].value_counts()
        rare_categories = dom_counts[dom_counts < rare_min_count].index
        processed = df['dom_element_type'].replace(rare_categories, 'Other')
        dom_unique = sorted(processed.unique())
        dom_mapping = {cat: idx for idx, cat in enumerate(dom_unique)}
        if 'Other' not in dom_mapping: dom_mapping['Other'] = len(dom_mapping)
        df['dom_element_type'] = processed.map(dom_mapping)
        learned['dom_mapping'] = dom_mapping
        learned['dom_rare_categories'] = set(rare_categories)
    elif 'dom_element_type' in df.columns:
        dom_mapping = learned.get('dom_mapping', {})
        rare_set = learned.get('dom_rare_categories', set())
        processed = df['dom_element_type'].replace(list(rare_set), 'Other')
        processed = processed.where(processed.isin(dom_mapping.keys()), 'Other')
        if 'Other' not in dom_mapping: dom_mapping['Other'] = max(dom_mapping.values(), default=-1) + 1
        df['dom_element_type'] = processed.map(dom_mapping)
    if save and out_path: df.to_csv(out_path, sep=";", index=False, encoding='utf-8')
    return df, learned

# -----------------------------
# Scaling
# -----------------------------
KINEMATIC_STD = ['cursor_speed', 'acceleration_mean', 'jitter','direction_changes', 'curvature_mean', 'rate_of_curvature']
PROPORTIONAL_MM = ['distance_travelled', 'movement_offset', 'straightness']
DURATION_STD = ['action_duration', 'time_since_last_action']
ROBUST_FEATS = ['self_intersections']
POS_X = ['x']
POS_Y = ['y']

def scalefeatures(df: pd.DataFrame, fit: bool = False, scalers: Optional[Dict] = None, clip_params: Optional[Dict] = None, drop_xy: bool = False, save: bool = False, out_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict, Dict]:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    def apply_clip(series: pd.Series, lower_val: float, upper_val: float) -> pd.Series:
        return series.clip(lower=lower_val, upper=upper_val)
    if fit:
        scalers = {} if scalers is None else scalers
        clip_params = {} if clip_params is None else clip_params
        if 'acceleration_mean' in df.columns:
            acc_ser = df['acceleration_mean'].dropna()
            clip_params['acc_lower'] = float(acc_ser.quantile(0.01))
            clip_params['acc_upper'] = float(acc_ser.quantile(0.99))
            df['acceleration_mean'] = apply_clip(df['acceleration_mean'], clip_params['acc_lower'], clip_params['acc_upper'])
        kin_mask = df['has_kinematics']==1 if 'has_kinematics' in df.columns and df['has_kinematics'].sum()>0 else pd.Series(True,index=df.index)
        kin_feats = [c for c in KINEMATIC_STD if c in df.columns]
        if kin_feats:
            kin_fit_rows = df.loc[kin_mask, kin_feats].dropna()
            if len(kin_fit_rows)<5: kin_fit_rows=df[kin_feats].dropna()
            scaler_kin=StandardScaler().fit(kin_fit_rows)
            scalers['kinematic_std']=scaler_kin
            df[kin_feats]=scaler_kin.transform(df[kin_feats].fillna(0))
        dur_feats=[c for c in DURATION_STD if c in df.columns]
        if dur_feats:
            dur_fit_rows=df.loc[df['has_duration']==1,dur_feats].dropna() if 'has_duration' in df.columns and df['has_duration'].sum()>0 else df[dur_feats].dropna()
            if len(dur_fit_rows)>0:
                scaler_dur=StandardScaler().fit(dur_fit_rows)
                scalers['duration_std']=scaler_dur
                df[dur_feats]=scaler_dur.transform(df[dur_feats].fillna(0))
        robust_feats=[c for c in ROBUST_FEATS if c in df.columns]
        if robust_feats:
            robust_fit_rows=df[robust_feats].dropna()
            if len(robust_fit_rows)>0:
                scaler_rb=RobustScaler().fit(robust_fit_rows)
                scalers['robust_counts']=scaler_rb
                df[robust_feats]=scaler_rb.transform(df[robust_feats].fillna(0))
        if not drop_xy and 'x' in df.columns:
            scaler_x=MinMaxScaler().fit(df[['x']].dropna())
            scalers['x_scaler']=scaler_x
            df['x']=scaler_x.transform(df[['x']].fillna(df[['x']].median()))
        if not drop_xy and 'y' in df.columns:
            scaler_y=RobustScaler().fit(df[['y']].dropna())
            scalers['y_scaler']=scaler_y
            df['y']=scaler_y.transform(df[['y']].fillna(df[['y']].median()))
        to_minmax=[c for c in PROPORTIONAL_MM if c in df.columns]
        if to_minmax:
            mm_fit_rows=df.loc[kin_mask,to_minmax].dropna()
            if len(mm_fit_rows)<5: mm_fit_rows=df[to_minmax].dropna()
            scaler_mm=MinMaxScaler(feature_range=(0,1)).fit(mm_fit_rows)
            scalers['block_minmax']=scaler_mm
            df[to_minmax]=scaler_mm.transform(df[to_minmax].fillna(0))
        if drop_xy:
            for col in ['x','y']: 
                if col in df.columns: df.drop(columns=[col], inplace=True)
    else:
        if clip_params and 'acc_lower' in clip_params and 'acc_upper' in clip_params and 'acceleration_mean' in df.columns:
            df['acceleration_mean']=apply_clip(df['acceleration_mean'], clip_params['acc_lower'], clip_params['acc_upper'])
        kin_feats=[c for c in KINEMATIC_STD if c in df.columns]
        if kin_feats and scalers and 'kinematic_std' in scalers:
            df[kin_feats]=scalers['kinematic_std'].transform(df[kin_feats].fillna(0))
        dur_feats=[c for c in DURATION_STD if c in df.columns]
        if dur_feats and scalers and 'duration_std' in scalers:
            df[dur_feats]=scalers['duration_std'].transform(df[dur_feats].fillna(0))
        robust_feats=[c for c in ROBUST_FEATS if c in df.columns]
        if robust_feats and scalers and 'robust_counts' in scalers:
            df[robust_feats]=scalers['robust_counts'].transform(df[robust_feats].fillna(0))
        if not drop_xy and 'x' in df.columns and scalers and 'x_scaler' in scalers:
            df['x']=scalers['x_scaler'].transform(df[['x']].fillna(df[['x']].median()))
        if not drop_xy and 'y' in df.columns and scalers and 'y_scaler' in scalers:
            df['y']=scalers['y_scaler'].transform(df[['y']].fillna(df[['y']].median()))
        to_minmax=[c for c in PROPORTIONAL_MM if c in df.columns]
        if to_minmax and scalers and 'block_minmax' in scalers:
            df[to_minmax]=scalers['block_minmax'].transform(df[to_minmax].fillna(0))
        if drop_xy:
            for col in ['x','y']:
                if col in df.columns: df.drop(columns=[col], inplace=True)
    if save and out_path: df.to_csv(out_path, sep=";", index=False, encoding='utf-8')
    return df, (scalers or {}), (clip_params or {})

# -----------------------------
# Sequence length plot
# -----------------------------
def plot_sequence_length_distribution(df: pd.DataFrame, session_col: str = "session_key"):
    seq_lengths=df.groupby(session_col).size()
    plt.figure(figsize=(10,6))
    sns.histplot(seq_lengths,bins=50,kde=True)
    plt.title("Sequence Length Distribution")
    plt.xlabel("Number of Actions in Session")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    print("Sequence Length Statistics:")
    print(seq_lengths.describe(percentiles=[.25,.5,.75,0.8,0.85,0.9,.95,.99]))


def feature_selection(df: pd.DataFrame, threshold: float = 0.9, target_column: Optional[str] = None):
    dropped_features = []
    df_copy = df.copy()

    features = df_copy.drop(columns=[target_column]) if target_column else df_copy
    features = features.select_dtypes(include=[float, int])

    if features.empty:
        print("No numeric columns found for correlation analysis.")
        return df_copy, []

    corr_matrix = features.corr().abs()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if to_drop:
        print(f"Dropping features due to correlation > {threshold}: {to_drop}")
    dropped_features = to_drop
    df_copy = df_copy.drop(columns=to_drop)

    return df_copy, dropped_features

# -----------------------------
# MAIN
# -----------------------------
if __name__=="__main__":
    train_csv_path = os.path.join("cv_outputs", "sessions_train.csv")
    report_path = os.path.join("cv_outputs", "data_summary_report_train.xlsx")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(
            f"Train split file not found: {train_csv_path}. "
            "Run data_preprocessing_participant_split.py first."
        )

    train_df = pd.read_csv(train_csv_path, delimiter=";", encoding="utf-8")

    # Leakage-safe EDA: report and sequence-length distribution on train split only.
    assess_data_quality(train_df, out_path=report_path)
    plot_sequence_length_distribution(train_df, session_col="session_key")
