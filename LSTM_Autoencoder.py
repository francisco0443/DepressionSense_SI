import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking,
    Embedding, Concatenate, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# Configuration / file paths
# -------------------------
base_file = "action_sequences_lstm_features.csv"  # raw/base
base_dir = os.path.dirname(base_file) or "."
train_file = os.path.join(base_dir, "action_sequences_lstm_features_train.csv")
val_file = os.path.join(base_dir, "action_sequences_lstm_features_val.csv")

# Categorical feature names (must match preprocessing outputs)
categorical_features = ["module_id", "action_type", "dom_element_type", "has_duration", "has_kinematics"]

DROP_XY_MODEL = False

# -------------------------
# Data Preparation
# -------------------------
def prepare_sequences(df_data, group_col="session_key", percentile_for_maxlen=80, maxlen=None):
    grouped = df_data.groupby(group_col)

    # infer continuous feature columns from DF (exclude session and categoricals)
    exclude_cols = [group_col] + categorical_features
    if DROP_XY_MODEL:
        exclude_cols += ['x', 'y']
    continuous_features = [c for c in df_data.columns if c not in exclude_cols]

    # build unpadded lists
    cont_seqs_unpadded = [g[continuous_features].values.astype(np.float32) for _, g in grouped]
    # categorical sequences: shift +1 to reserve 0 for pad
    cat_seqs_unpadded = {
        f: [g[f].values.astype(np.int32) + 1 for _, g in grouped]
        for f in categorical_features
    }

    # compute maxlen from train if not provided
    if maxlen is None:
        seq_lengths = [len(s) for s in cont_seqs_unpadded]
        maxlen = int(np.percentile(seq_lengths, percentile_for_maxlen))
        if maxlen < 1:
            maxlen = max(seq_lengths) if seq_lengths else 1
        print(f"✅ Max sequence length ({percentile_for_maxlen}th percentile): {maxlen}")
    else:
        print(f"✅ Using provided maxlen = {maxlen}")

    # pad continuous with 0.0 (padding sentinel)
    cont_seqs_padded = pad_sequences(cont_seqs_unpadded, maxlen=maxlen, dtype="float32", value=0.0)

    # pad categorical with 0 (padding id)
    cat_seqs_padded = {}
    for f in categorical_features:
        cat_seqs_padded[f] = pad_sequences(cat_seqs_unpadded[f], maxlen=maxlen, dtype="int32", value=0)
        u = np.unique(cat_seqs_padded[f])
        print(f"✅ '{f}' IDs min..max after shift+pad: {u.min()}..{u.max()} (0 is pad)")

    return cont_seqs_padded, cat_seqs_padded, maxlen, continuous_features


# -------------------------
# Model builder
# -------------------------
def build_lstm_autoencoder(timesteps, num_cont_features, categorical_features_list, embedding_cfg):
    cont_input = Input(shape=(timesteps, num_cont_features), name="continuous_input")

    cat_inputs, cat_embeds = [], []
    for f in categorical_features_list:
        inp = Input(shape=(timesteps,), dtype="int32", name=f)
        if f not in embedding_cfg:
            raise ValueError(f"Missing embedding config for categorical '{f}'")
        input_dim, output_dim = embedding_cfg[f]

        emb_raw = Embedding(input_dim=input_dim, output_dim=output_dim, mask_zero=False,
                            name=f"{f}_embedding")(inp)

        pad_mask = Lambda(lambda z: tf.cast(tf.not_equal(z, 0), tf.float32)[..., None], name=f"{f}_padmask")(inp)
        emb = Lambda(lambda args: args[0] * args[1], name=f"{f}_masked_emb")([emb_raw, pad_mask])

        cat_inputs.append(inp)
        cat_embeds.append(emb)

    concat = Concatenate(axis=-1, name="concat_inputs")([cont_input] + cat_embeds)
    masked = Masking(mask_value=0.0, name="mask_after_concat")(concat)

    encoded = LSTM(32, activation="tanh", dropout=0.2, name="encoder_lstm")(masked)

    bottleneck = Dense(8, activation='tanh', name='bottleneck',
                       kernel_regularizer=tf.keras.regularizers.l2(1e-6))(encoded)

    repeated = RepeatVector(timesteps, name="repeat")(bottleneck)
    decoded = LSTM(32, activation="tanh", return_sequences=True, dropout=0.2,
                   name="decoder_lstm")(repeated)

    cont_out = TimeDistributed(Dense(num_cont_features), name="reconstructed_continuous")(decoded)

    model = Model(inputs=[cont_input] + cat_inputs, outputs=cont_out)
    model.compile(optimizer=Adam(learning_rate=1e-2), loss="mse")
    return model


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    if os.path.exists(train_file) and os.path.exists(val_file):
        print("✅ Found preprocessed train/val CSVs. Loading them and respecting the session-level split.")
        train_df = pd.read_csv(train_file, delimiter=';')
        val_df = pd.read_csv(val_file, delimiter=';')

        train_sessions = set(train_df['session_key'].astype(str).unique())
        val_sessions = set(val_df['session_key'].astype(str).unique())
        print("train sessions:", len(train_sessions), "val sessions:", len(val_sessions),
              "intersection:", len(train_sessions & val_sessions))

        cont_tr, cat_tr, timesteps, continuous_features = prepare_sequences(train_df, group_col="session_key",
                                                                            percentile_for_maxlen=95, maxlen=None)
        
    

        cont_va, cat_va, _, _ = prepare_sequences(val_df, group_col="session_key", maxlen=timesteps)

        print(f"\nModel will use {len(continuous_features)} continuous features and {timesteps} timesteps (from TRAIN).")

        embedding_config = {}
        for f in categorical_features:
            max_id = int(np.max(cat_tr[f]))  # max value after shift+pad
            input_dim = max_id + 1  # include 0 pad id
            cardinality = max(0, input_dim - 1)
            emb_dim = min(50, max(1, (cardinality // 2) + 1))
            embedding_config[f] = (input_dim, emb_dim)
            print(f"Embedding config for {f}: input_dim={input_dim}, emb_dim={emb_dim}")

        X_cont_tr, X_cont_va = cont_tr, cont_va
        X_cat_tr = [cat_tr[f] for f in categorical_features]
        X_cat_va = [cat_va[f] for f in categorical_features]
        y_tr, y_va = X_cont_tr, X_cont_va

    else:
        print("⚠️ Preprocessed train/val CSVs not found. Falling back to single-file behavior (random split).")
        df = pd.read_csv(base_file, delimiter=';')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        cont_all, cat_all, timesteps, continuous_features = prepare_sequences(df, group_col="session_key",
                                                                              percentile_for_maxlen=95)
        num_samples = len(cont_all)
        perm = np.random.permutation(num_samples)
        split = int(0.8 * num_samples)
        tr_idx, va_idx = perm[:split], perm[split:]

        X_cont_tr, X_cont_va = cont_all[tr_idx], cont_all[va_idx]
        X_cat_tr = [cat_all[f][tr_idx] for f in categorical_features]
        X_cat_va = [cat_all[f][va_idx] for f in categorical_features]
        y_tr, y_va = X_cont_tr, X_cont_va

        embedding_config = {}
        for f in categorical_features:
            max_id = int(np.max(X_cat_tr[categorical_features.index(f)]))
            input_dim = max_id + 1
            cardinality = max(0, input_dim - 1)
            emb_dim = min(50, max(1, (cardinality // 2) + 1))
            embedding_config[f] = (input_dim, emb_dim)
            print(f"Embedding config for {f}: input_dim={input_dim}, emb_dim={emb_dim}")

        print(f"\nModel will use {len(continuous_features)} continuous features and {timesteps} timesteps.")

    # Build model
    num_cont_features = X_cont_tr.shape[2]
    model = build_lstm_autoencoder(
        timesteps=timesteps,
        num_cont_features=num_cont_features,
        categorical_features_list=categorical_features,
        embedding_cfg=embedding_config
    )
    model.summary()

    n_tr = X_cont_tr.shape[0]
    n_va = X_cont_va.shape[0]
    print(f"\nTraining on {n_tr} samples, validating on {n_va} samples.")

    # -------------------------
    # Checkpoint: save weights-only using Keras weights format suffix
    # -------------------------
    checkpoint_filepath = 'best_model.weights.h5'   # MUST end with .weights.h5 when save_weights_only=True

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            save_weights_only=True,    # <<-- important: weights-only to avoid serializing Lambdas
            monitor='val_loss',
            mode='min',
            verbose=1
        )
    ]

    # Build boolean masks: True for real timesteps, False for padding (use action_type explicitly)
    act_idx = categorical_features.index("action_type")
    mask_tr = (X_cat_tr[act_idx] != 0)
    mask_va = (X_cat_va[act_idx] != 0)
    w_tr = mask_tr.astype('float32')
    w_va = mask_va.astype('float32')

    history = model.fit(
        [X_cont_tr] + X_cat_tr, y_tr,
        sample_weight=w_tr,
        validation_data=([X_cont_va] + X_cat_va, y_va, w_va),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        verbose=2
    )

    print("\n✅ Training complete!")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    if 'val_loss' in history.history:
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    # ensure final weights saved (ModelCheckpoint already saved best weights)
    model.save_weights(checkpoint_filepath)

    # save minimal metadata needed to rebuild architecture & preprocessing mapping
    metadata = {
        "timesteps": int(timesteps),
        "continuous_features": continuous_features,
        "categorical_features": categorical_features,
        "embedding_config": {f: embedding_config[f] for f in embedding_config},
        "lstm_units": 32,
        "bottleneck_dim": 8,
        "dropout": 0.2,
        "drop_xy_model": bool(DROP_XY_MODEL),
        "random_seed": 42
    }

    with open("best_model_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print("Saved weights to", checkpoint_filepath, "and metadata to best_model_metadata.json")
