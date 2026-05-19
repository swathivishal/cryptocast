"""
=============================================================
 CryptoCast: Multi-Horizon Bitcoin Price Forecasting
 Using Deep Learning — Modular Programming Version
=============================================================
 Domain  : Financial Analytics / Cryptocurrency / Deep Learning
 Tasks   :
   Task 1 - Supervised (Regression) : 1-Day, 3-Day, 7-Day
           Bitcoin Price Forecasting
   Task 2 - Model Comparison        : 1D-CNN, RNN, LSTM,
           Transformer
=============================================================
"""

# ─── Imports ─────────────────────────────────────────────────────
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv1D, MaxPooling1D,
                                     Flatten, SimpleRNN, LSTM,
                                     Dropout, Input, MultiHeadAttention,
                                     LayerNormalization, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# ─── Global Plot Config ─────────────────────────────────────────
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 12

# ─── Suppress TF Warnings ──────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ─── Reproducibility ────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Output Directory ───────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Flush Print Helper ─────────────────────────────────────────
def log(msg: str) -> None:
    """Print with immediate flush for real-time logging."""
    print(msg, flush=True)


def save_path(filename: str) -> str:
    """Return full path for output file in script directory."""
    return os.path.join(OUTPUT_DIR, filename)


# ══════════════════════════════════════════════════════════════════
# MODULE 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Bitcoin historical price dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(filepath)
    log(f"✅ Data Loaded | Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════
# MODULE 2 — DATA CLEANING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def parse_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse string-formatted numeric columns to float.
    Handles commas, percentage signs, and K/M/B suffixes in Volume.

    Args:
        df (pd.DataFrame): Raw DataFrame with string columns.

    Returns:
        pd.DataFrame: DataFrame with parsed numeric columns.
    """
    price_cols = ["Price", "Open", "High", "Low"]

    for col in price_cols:
        df[col] = (df[col].astype(str)
                   .str.replace(",", "", regex=False)
                   .astype(float))

    # Parse Volume: remove commas, handle K/M/B suffixes
    def parse_volume(val: str) -> float:
        val = str(val).replace(",", "").strip()
        if val.endswith("K"):
            return float(val[:-1]) * 1_000
        elif val.endswith("M"):
            return float(val[:-1]) * 1_000_000
        elif val.endswith("B"):
            return float(val[:-1]) * 1_000_000_000
        try:
            return float(val)
        except ValueError:
            return np.nan

    df["Vol."] = df["Vol."].apply(parse_volume)

    # Parse Change %: remove % sign
    df["Change %"] = (df["Change %"].astype(str)
                      .str.replace("%", "", regex=False)
                      .astype(float))

    log(f"✅ Numeric Columns Parsed | Dtypes:\n{df.dtypes}")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Date column to datetime and sort chronologically (ascending).

    Args:
        df (pd.DataFrame): DataFrame with string Date column.

    Returns:
        pd.DataFrame: DataFrame with parsed datetime, sorted ascending.
    """
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)
    log(f"✅ Dates Parsed & Sorted | Range: "
        f"{df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")
    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """Print missing value counts for each column."""
    missing = df.isnull().sum()
    if missing.sum() == 0:
        log("✅ No missing values found.")
    else:
        log("⚠️  Missing Values:\n" + str(missing[missing > 0]))


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using forward-fill then backward-fill.

    Args:
        df (pd.DataFrame): DataFrame with potential NaNs.

    Returns:
        pd.DataFrame: DataFrame with no missing values.
    """
    before = df.isnull().sum().sum()
    df = df.ffill().bfill()
    after = df.isnull().sum().sum()
    log(f"✅ Missing Values Handled | Before: {before} | After: {after}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline: parse numerics, parse dates,
    check & handle missing values.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    log("\n--- MODULE 2: DATA CLEANING & PREPROCESSING ---")
    df = parse_numeric_columns(df)
    df = parse_dates(df)
    check_missing_values(df)
    df = handle_missing_values(df)
    log(f"✅ Data Cleaned | Final Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════
# MODULE 3 — EDA (Exploratory Data Analysis)
# ══════════════════════════════════════════════════════════════════

def plot_price_trend(df: pd.DataFrame) -> None:
    """Plot Bitcoin closing price over time."""
    plt.figure(figsize=(14, 6))
    plt.plot(df["Date"], df["Price"], color="steelblue", lw=1.5)
    plt.title("Bitcoin Closing Price Over Time",
              fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(save_path("eda_price_trend.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("✅ Saved: eda_price_trend.png")


def plot_volume_trend(df: pd.DataFrame) -> None:
    """Plot Bitcoin trading volume over time."""
    plt.figure(figsize=(14, 5))
    plt.plot(df["Date"], df["Vol."], color="darkorange", lw=1, alpha=0.7)
    plt.title("Bitcoin Trading Volume Over Time",
              fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.savefig(save_path("eda_volume_trend.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("✅ Saved: eda_volume_trend.png")


def plot_price_distribution(df: pd.DataFrame) -> None:
    """Plot histogram and boxplot for Bitcoin price."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Price"], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Distribution of Bitcoin Price")
    sns.boxplot(x=df["Price"], ax=axes[1], color="lightcoral")
    axes[1].set_title("Boxplot of Bitcoin Price")
    plt.suptitle("Bitcoin Price Distribution",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path("eda_price_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("✅ Saved: eda_price_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, linewidths=0.5)
    plt.title("Feature Correlation Heatmap",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path("eda_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("✅ Saved: eda_correlation_heatmap.png")


def plot_daily_change(df: pd.DataFrame) -> None:
    """Plot daily percentage change distribution."""
    plt.figure(figsize=(12, 5))
    sns.histplot(df["Change %"], kde=True, color="teal", bins=80)
    plt.axvline(0, color="red", linestyle="--", lw=1.5, label="0%")
    plt.title("Daily % Change Distribution",
              fontsize=14, fontweight="bold")
    plt.xlabel("Change %")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path("eda_daily_change.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("✅ Saved: eda_daily_change.png")


def plot_ohlc_comparison(df: pd.DataFrame) -> None:
    """Plot Open, High, Low, Close price comparison (last 365 days)."""
    recent = df.tail(365)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(recent["Date"], recent["Price"], label="Close", lw=2, color="steelblue")
    ax.plot(recent["Date"], recent["Open"],  label="Open",  lw=1, alpha=0.6, color="green")
    ax.plot(recent["Date"], recent["High"],  label="High",  lw=1, alpha=0.4, color="red")
    ax.plot(recent["Date"], recent["Low"],   label="Low",   lw=1, alpha=0.4, color="orange")
    ax.set_title("OHLC Price Comparison (Last 365 Days)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path("eda_ohlc_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("✅ Saved: eda_ohlc_comparison.png")


def run_eda(df: pd.DataFrame) -> None:
    """
    Run full EDA: price trend, volume, distribution,
    correlation, daily change, OHLC comparison.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.
    """
    log("\n--- MODULE 3: EDA ---")
    plot_price_trend(df)
    plot_volume_trend(df)
    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_daily_change(df)
    plot_ohlc_comparison(df)
    log("✅ EDA Complete | All plots saved.")


# ══════════════════════════════════════════════════════════════════
# MODULE 4 — FEATURE ENGINEERING & SCALING
# ══════════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame,
                   features: list = None) -> pd.DataFrame:
    """
    Select and return the feature columns for modeling.

    Args:
        df       (pd.DataFrame): Cleaned DataFrame.
        features (list)        : List of column names to use.
                                 Default: all numeric except Date.

    Returns:
        pd.DataFrame: DataFrame with selected features.
    """
    if features is None:
        features = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    existing = [f for f in features if f in df.columns]
    df_features = df[existing].copy()
    log(f"✅ Features Selected: {existing}")
    return df_features


def scale_features(df_features: pd.DataFrame) -> tuple:
    """
    Apply MinMaxScaler to feature matrix.
    Saves the scaler for inverse_transform later.

    Args:
        df_features (pd.DataFrame): Feature DataFrame.

    Returns:
        tuple: (scaled_data as np.ndarray, scaler object)
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)
    log(f"✅ MinMax Scaling Applied | Shape: {scaled_data.shape} | "
        f"Range: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")
    return scaled_data, scaler


def run_feature_engineering(df: pd.DataFrame,
                            features: list = None) -> tuple:
    """
    Full feature engineering pipeline: select & scale.

    Args:
        df       (pd.DataFrame): Cleaned DataFrame.
        features (list)        : Feature columns to use.

    Returns:
        tuple: (scaled_data, scaler, feature_names)
    """
    log("\n--- MODULE 4: FEATURE ENGINEERING & SCALING ---")
    if features is None:
        features = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    df_features = select_features(df, features)
    scaled_data, scaler = scale_features(df_features)
    return scaled_data, scaler, features


# ══════════════════════════════════════════════════════════════════
# MODULE 5 — SEQUENCE GENERATION (Sliding Window)
# ══════════════════════════════════════════════════════════════════

def create_sequences(data: np.ndarray,
                     seq_length: int = 60,
                     horizons: list = None) -> dict:
    """
    Generate sliding-window sequences for multi-horizon forecasting.

    For each window of `seq_length` past days, creates targets for
    1-day, 3-day, and 7-day ahead forecasts using the Close price
    (column index 0).

    Args:
        data       (np.ndarray): Scaled feature matrix (samples, features).
        seq_length (int)       : Number of past days per input sequence.
        horizons   (list)      : List of forecast horizons in days.

    Returns:
        dict: {
            "X"        : np.ndarray (samples, seq_length, features),
            "y_1d"     : np.ndarray (samples,),
            "y_3d"     : np.ndarray (samples,),
            "y_7d"     : np.ndarray (samples,),
            "horizons" : list
        }
    """
    if horizons is None:
        horizons = [1, 3, 7]

    X_list = []
    y_dict = {h: [] for h in horizons}

    max_horizon = max(horizons)

    for i in range(len(data) - seq_length - max_horizon + 1):
        X_list.append(data[i : i + seq_length])
        for h in horizons:
            # Target is Close price (index 0) at horizon steps ahead
            y_dict[h].append(data[i + seq_length + h - 1, 0])

    X = np.array(X_list)
    y_arrays = {f"y_{h}d": np.array(y_dict[h]) for h in horizons}

    log(f"✅ Sequences Generated | seq_length: {seq_length}")
    log(f"   X shape: {X.shape}")
    for key, arr in y_arrays.items():
        log(f"   {key} shape: {arr.shape}")

    return {"X": X, **y_arrays, "horizons": horizons}


def split_sequences(seq_data: dict,
                    test_ratio: float = 0.2) -> dict:
    """
    Time-based train-test split (no shuffling to avoid data leakage).

    Args:
        seq_data   (dict): Output from create_sequences().
        test_ratio (float): Proportion of data for testing.

    Returns:
        dict: Train-test split data for all horizons.
    """
    n = len(seq_data["X"])
    split_idx = int(n * (1 - test_ratio))

    result = {
        "X_train": seq_data["X"][:split_idx],
        "X_test" : seq_data["X"][split_idx:],
        "horizons": seq_data["horizons"],
    }

    for h in seq_data["horizons"]:
        key = f"y_{h}d"
        result[f"y_train_{h}d"] = seq_data[key][:split_idx]
        result[f"y_test_{h}d"]  = seq_data[key][split_idx:]

    log(f"✅ Train-Test Split | Train: {split_idx} | Test: {n - split_idx}")
    log(f"   X_train: {result['X_train'].shape} | X_test: {result['X_test'].shape}")
    return result


def run_sequence_generation(scaled_data: np.ndarray,
                            seq_length: int = 60,
                            test_ratio: float = 0.2) -> dict:
    """
    Full sequence generation pipeline: create & split.

    Args:
        scaled_data (np.ndarray): Scaled feature matrix.
        seq_length  (int)       : Input sequence length.
        test_ratio  (float)     : Test set proportion.

    Returns:
        dict: Complete split data with sequences.
    """
    log("\n--- MODULE 5: SEQUENCE GENERATION ---")
    seq_data = create_sequences(scaled_data, seq_length)
    split_data = split_sequences(seq_data, test_ratio)
    return split_data


# ══════════════════════════════════════════════════════════════════
# MODULE 6 — MODEL BUILDING
# ══════════════════════════════════════════════════════════════════

def build_cnn(input_shape: tuple) -> Model:
    """
    Build 1D-CNN model for time-series forecasting.
    Captures local temporal patterns with fast training.

    Args:
        input_shape (tuple): (seq_length, n_features).

    Returns:
        Model: Compiled Keras model.
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation="relu",
               input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ], name="1D_CNN")
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model


def build_rnn(input_shape: tuple) -> Model:
    """
    Build Simple RNN model for time-series forecasting.
    Baseline temporal model with sequential dependency modeling.

    Args:
        input_shape (tuple): (seq_length, n_features).

    Returns:
        Model: Compiled Keras model.
    """
    model = Sequential([
        SimpleRNN(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        SimpleRNN(16, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ], name="Simple_RNN")
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model


def build_lstm(input_shape: tuple) -> Model:
    """
    Build LSTM model for time-series forecasting.
    Handles long-term dependencies, reduces vanishing gradient.

    Args:
        input_shape (tuple): (seq_length, n_features).

    Returns:
        Model: Compiled Keras model.
    """
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ], name="LSTM")
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model


def build_transformer(input_shape: tuple,
                      head_size: int = 32,
                      num_heads: int = 2,
                      ff_dim: int = 64,
                      num_transformer_blocks: int = 1) -> Model:
    """
    Build Transformer model for time-series forecasting.
    Uses self-attention mechanism to capture global dependencies.

    Args:
        input_shape             (tuple): (seq_length, n_features).
        head_size               (int)  : Size of each attention head.
        num_heads               (int)  : Number of attention heads.
        ff_dim                  (int)  : Feed-forward dimension.
        num_transformer_blocks  (int)  : Number of transformer blocks.

    Returns:
        Model: Compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    for _ in range(num_transformer_blocks):
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size
        )(x, x)
        attn_output = Dropout(0.1)(attn_output)
        x1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-Forward Network
        ff_output = Dense(ff_dim, activation="relu")(x1)
        ff_output = Dense(input_shape[-1])(ff_output)
        ff_output = Dropout(0.1)(ff_output)
        x = LayerNormalization(epsilon=1e-6)(x1 + ff_output)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs, name="Transformer")
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model


def get_models(input_shape: tuple) -> dict:
    """
    Return dictionary of all deep learning models.

    Args:
        input_shape (tuple): (seq_length, n_features).

    Returns:
        dict: {model_name: compiled_model}
    """
    return {
        "1D-CNN"      : build_cnn(input_shape),
        "RNN"         : build_rnn(input_shape),
        "LSTM"        : build_lstm(input_shape),
        "Transformer" : build_transformer(input_shape),
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 7 — MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════

def train_model(model: Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                epochs: int = 30,
                batch_size: int = 64) -> tuple:
    """
    Train a Keras model with early stopping.

    Args:
        model      (Model)      : Compiled Keras model.
        X_train    (np.ndarray) : Training features.
        y_train    (np.ndarray) : Training targets.
        X_val      (np.ndarray) : Validation features.
        y_val      (np.ndarray) : Validation targets.
        epochs     (int)        : Maximum training epochs.
        batch_size (int)        : Mini-batch size.

    Returns:
        tuple: (trained_model, training_history)
    """
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5,
        restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    return model, history


def evaluate_model(model: Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   scaler: MinMaxScaler) -> dict:
    """
    Evaluate a trained model on test data.
    Inverse-transforms predictions to original price scale.

    Args:
        model  (Model)        : Trained Keras model.
        X_test (np.ndarray)   : Test features.
        y_test (np.ndarray)   : True target values (scaled).
        scaler (MinMaxScaler) : Fitted scaler for inverse transform.

    Returns:
        dict: MAE, RMSE, MAPE, y_pred_inv, y_test_inv
    """
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform: create dummy array with n_features columns
    n_features = scaler.n_features_in_
    y_pred_full = np.zeros((len(y_pred_scaled), n_features))
    y_test_full = np.zeros((len(y_test), n_features))
    y_pred_full[:, 0] = y_pred_scaled
    y_test_full[:, 0] = y_test

    y_pred_inv = scaler.inverse_transform(y_pred_full)[:, 0]
    y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]

    mae  = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) /
                          (y_test_inv + 1e-10))) * 100

    return {
        "MAE"        : round(mae, 4),
        "RMSE"       : round(rmse, 4),
        "MAPE"       : round(mape, 4),
        "y_pred_inv" : y_pred_inv,
        "y_test_inv" : y_test_inv,
    }


def train_and_evaluate_all(split_data: dict,
                           scaler: MinMaxScaler,
                           epochs: int = 30,
                           batch_size: int = 64) -> dict:
    """
    Train and evaluate all models across all horizons.

    Args:
        split_data (dict)        : Output from split_sequences().
        scaler     (MinMaxScaler): Fitted scaler.
        epochs     (int)         : Training epochs.
        batch_size (int)         : Batch size.

    Returns:
        dict: {
            "results_df"    : pd.DataFrame with all metrics,
            "trained_models": {tag: {model, history, y_pred, y_test}},
            "all_results"   : list of result dicts
        }
    """
    log("\n--- MODULE 7: MODEL TRAINING & EVALUATION ---")

    X_train = split_data["X_train"]
    X_test  = split_data["X_test"]
    input_shape = (X_train.shape[1], X_train.shape[2])
    horizons = split_data["horizons"]

    # Use last 10% of training as validation
    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]

    all_results = []
    trained_models = {}

    for h in horizons:
        log(f"\n{'='*50}")
        log(f"  HORIZON: {h}-Day Forecast")
        log(f"{'='*50}")

        y_train_h = split_data[f"y_train_{h}d"]
        y_test_h  = split_data[f"y_test_{h}d"]
        y_tr, y_val_h = y_train_h[:val_split], y_train_h[val_split:]

        models = get_models(input_shape)

        for name, model in models.items():
            tag = f"{name}_{h}D"
            log(f"  ⏳ Training: {tag}...")
            log(f"     Params: {model.count_params():,}")

            trained_model, history = train_model(
                model, X_tr, y_tr, X_val, y_val_h,
                epochs=epochs, batch_size=batch_size
            )

            metrics = evaluate_model(trained_model, X_test, y_test_h, scaler)

            result = {
                "Model"    : name,
                "Horizon"  : f"{h}D",
                "MAE"      : metrics["MAE"],
                "RMSE"     : metrics["RMSE"],
                "MAPE"     : metrics["MAPE"],
            }
            all_results.append(result)

            trained_models[tag] = {
                "model"    : trained_model,
                "history"  : history,
                "y_pred"   : metrics["y_pred_inv"],
                "y_test"   : metrics["y_test_inv"],
            }

            log(f"  ✅ {tag} Done | "
                  f"MAE: {metrics['MAE']:.2f} | "
                  f"RMSE: {metrics['RMSE']:.2f} | "
                  f"MAPE: {metrics['MAPE']:.2f}%")

            # Free memory
            del trained_model

    results_df = pd.DataFrame(all_results)
    log(f"\n✅ All Models Trained & Evaluated!")
    log(results_df.to_string(index=False))

    return {
        "results_df"    : results_df,
        "trained_models": trained_models,
        "all_results"   : all_results,
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 8 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def plot_loss_curves(trained_models: dict) -> None:
    """
    Plot training vs validation loss curves for all models.

    Args:
        trained_models (dict): Trained model results dictionary.
    """
    tags = list(trained_models.keys())
    n = len(tags)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, tag in enumerate(tags):
        history = trained_models[tag]["history"]
        axes[idx].plot(history.history["loss"],
                       label="Train Loss", color="steelblue", lw=1.5)
        axes[idx].plot(history.history["val_loss"],
                       label="Val Loss", color="darkorange", lw=1.5)
        axes[idx].set_title(f"Loss Curve — {tag}", fontweight="bold")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Loss (MSE)")
        axes[idx].legend(loc="best")
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Training & Validation Loss Curves",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path("viz_loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("✅ Saved: viz_loss_curves.png")


def plot_actual_vs_predicted(trained_models: dict,
                             horizons: list = None) -> None:
    """
    Plot actual vs predicted price curves for each model & horizon.

    Args:
        trained_models (dict): Trained model results.
        horizons       (list): Forecast horizons.
    """
    if horizons is None:
        horizons = [1, 3, 7]

    for h in horizons:
        model_names = ["1D-CNN", "RNN", "LSTM", "Transformer"]
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        for idx, name in enumerate(model_names):
            tag = f"{name}_{h}D"
            if tag not in trained_models:
                axes[idx].set_visible(False)
                continue

            y_test = trained_models[tag]["y_test"]
            y_pred = trained_models[tag]["y_pred"]

            axes[idx].plot(y_test, label="Actual",
                           color="steelblue", lw=1.5)
            axes[idx].plot(y_pred, label="Predicted",
                           color="darkorange", lw=1.2, alpha=0.8)
            axes[idx].set_title(f"{name} — {h}-Day Forecast",
                                fontweight="bold")
            axes[idx].set_xlabel("Time Step")
            axes[idx].set_ylabel("Price (USD)")
            axes[idx].legend(loc="best")
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle(f"Actual vs Predicted — {h}-Day Horizon",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path(f"viz_actual_vs_pred_{h}D.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        log(f"✅ Saved: viz_actual_vs_pred_{h}D.png")


def plot_error_distribution(trained_models: dict,
                            horizons: list = None) -> None:
    """
    Plot error distribution (residuals) for each model & horizon.

    Args:
        trained_models (dict): Trained model results.
        horizons       (list): Forecast horizons.
    """
    if horizons is None:
        horizons = [1, 3, 7]

    for h in horizons:
        model_names = ["1D-CNN", "RNN", "LSTM", "Transformer"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        for idx, name in enumerate(model_names):
            tag = f"{name}_{h}D"
            if tag not in trained_models:
                axes[idx].set_visible(False)
                continue

            y_test = trained_models[tag]["y_test"]
            y_pred = trained_models[tag]["y_pred"]
            errors = y_test - y_pred

            sns.histplot(errors, kde=True, ax=axes[idx],
                         color="teal", bins=50)
            axes[idx].axvline(0, color="red", linestyle="--", lw=1.5)
            axes[idx].set_title(f"{name} — Error Distribution ({h}D)",
                                fontweight="bold")
            axes[idx].set_xlabel("Prediction Error (USD)")

        plt.suptitle(f"Error Distribution — {h}-Day Horizon",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path(f"viz_error_dist_{h}D.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        log(f"✅ Saved: viz_error_dist_{h}D.png")


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """
    Bar chart comparing all models across horizons on MAE, RMSE, MAPE.

    Args:
        results_df (pd.DataFrame): Model comparison table.
    """
    metrics = ["MAE", "RMSE", "MAPE"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, metric in zip(axes, metrics):
        sns.barplot(data=results_df, x="Model", y=metric,
                    hue="Horizon", ax=ax, palette="Set2")
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("Model")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Performance Comparison Across Horizons",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path("viz_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("✅ Saved: viz_model_comparison.png")


def plot_horizon_comparison(results_df: pd.DataFrame) -> None:
    """
    Line chart showing how each model's performance degrades across horizons.

    Args:
        results_df (pd.DataFrame): Model comparison table.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
        for model_name in results_df["Model"].unique():
            subset = results_df[results_df["Model"] == model_name]
            ax.plot(subset["Horizon"], subset[metric],
                    marker="o", lw=2, label=model_name)
        ax.set_title(f"{metric} by Horizon", fontweight="bold")
        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel(metric)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Forecast Horizon vs Performance",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path("viz_horizon_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("✅ Saved: viz_horizon_comparison.png")


def plot_scatter_actual_vs_pred(trained_models: dict,
                                horizons: list = None) -> None:
    """
    Scatter plot of actual vs predicted for best model per horizon.

    Args:
        trained_models (dict): Trained model results.
        horizons       (list): Forecast horizons.
    """
    if horizons is None:
        horizons = [1, 3, 7]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, h in enumerate(horizons):
        # Use LSTM as representative model for scatter
        tag = f"LSTM_{h}D"
        if tag not in trained_models:
            tag = list(trained_models.keys())[0]

        y_test = trained_models[tag]["y_test"]
        y_pred = trained_models[tag]["y_pred"]

        axes[idx].scatter(y_test, y_pred, alpha=0.3, s=10,
                          color="steelblue")
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val],
                       "r--", lw=2, label="Perfect Prediction")
        axes[idx].set_xlabel("Actual Price (USD)")
        axes[idx].set_ylabel("Predicted Price (USD)")
        axes[idx].set_title(f"LSTM — {h}-Day Forecast", fontweight="bold")
        axes[idx].legend(loc="best")
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle("Scatter: Actual vs Predicted (LSTM)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path("viz_scatter_actual_vs_pred.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    log("✅ Saved: viz_scatter_actual_vs_pred.png")


def run_visualization(results_df: pd.DataFrame,
                      trained_models: dict,
                      horizons: list = None) -> None:
    """
    Full visualization pipeline: loss curves, actual vs predicted,
    error distribution, model comparison, horizon comparison,
    scatter plots.

    Args:
        results_df     (pd.DataFrame): Model comparison table.
        trained_models (dict)        : Trained model results.
        horizons       (list)        : Forecast horizons.
    """
    if horizons is None:
        horizons = [1, 3, 7]

    log("\n--- MODULE 8: VISUALIZATION ---")
    plot_loss_curves(trained_models)
    plot_actual_vs_predicted(trained_models, horizons)
    plot_error_distribution(trained_models, horizons)
    plot_model_comparison(results_df)
    plot_horizon_comparison(results_df)
    plot_scatter_actual_vs_pred(trained_models, horizons)
    log("✅ All Visualizations Complete | All plots saved.")


# ══════════════════════════════════════════════════════════════════
# MODULE 9 — SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════

def print_summary(results_df: pd.DataFrame) -> None:
    """
    Print final project summary report.

    Args:
        results_df (pd.DataFrame): Complete model comparison table.
    """
    log("\n" + "=" * 70)
    log("       CRYPTOCAST — FINAL RESULTS SUMMARY")
    log("=" * 70)

    log("\n📌 PIPELINE:")
    pipeline_steps = [
        "Data Loading & Cleaning",
        "Feature Engineering (MinMax Scaling)",
        "Sequence Generation (Sliding Window, 60 days)",
        "Time-Based Train-Test Split (80/20)",
        "Model Building (1D-CNN, RNN, LSTM, Transformer)",
        "Multi-Horizon Forecasting (1D, 3D, 7D)",
        "Evaluation (MAE, RMSE, MAPE)",
    ]
    for i, step in enumerate(pipeline_steps, 1):
        log(f"  {i}. {step}")

    log("\n📊 FULL MODEL COMPARISON:")
    log(results_df.to_string(index=False))

    # Best model per horizon
    log("\n🏆 Best Models Per Horizon:")
    for h in ["1D", "3D", "7D"]:
        subset = results_df[results_df["Horizon"] == h]
        if len(subset) > 0:
            best = subset.loc[subset["RMSE"].idxmin()]
            log(f"   {h} Forecast → {best['Model']}: "
                f"MAE={best['MAE']:.2f}, RMSE={best['RMSE']:.2f}, "
                f"MAPE={best['MAPE']:.2f}%")

    # Overall best
    best_overall = results_df.loc[results_df["RMSE"].idxmin()]
    log(f"\n🥇 Overall Best: {best_overall['Model']} "
        f"({best_overall['Horizon']}) — "
        f"RMSE={best_overall['RMSE']:.2f}")

    log("\n✅ All charts saved as PNG files in project folder!")
    log("=" * 70)


# ══════════════════════════════════════════════════════════════════
# MAIN — RUN FULL PIPELINE
# ══════════════════════════════════════════════════════════════════

def main():
    """
    Main function — orchestrates the full CryptoCast pipeline.
    Run this file directly: python cryptocast_modular.py
    """

    # ── CONFIG ──────────────────────────────────────────────────
    DATA_PATH    = os.path.join(OUTPUT_DIR, "..", "upload",
                                "crypto_dataset",
                                "Bitcoin_Historical_Data.csv")
    FEATURES     = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    SEQ_LENGTH   = 60
    HORIZONS     = [1, 3, 7]
    TEST_RATIO   = 0.2
    EPOCHS       = 30
    BATCH_SIZE   = 64
    SEED_VAL     = 42

    # ── Set Seeds ───────────────────────────────────────────────
    np.random.seed(SEED_VAL)
    tf.random.set_seed(SEED_VAL)

    log("=" * 60)
    log("   CRYPTOCAST: MULTI-HORIZON BITCOIN PRICE")
    log("   FORECASTING USING DEEP LEARNING")
    log("   Modular Python Version")
    log("=" * 60)

    # ── MODULE 1: LOAD DATA ─────────────────────────────────────
    log("\n--- MODULE 1: DATA LOADING ---")
    df = load_data(DATA_PATH)

    # ── MODULE 2: CLEAN DATA ────────────────────────────────────
    df = clean_data(df)

    # ── MODULE 3: EDA ───────────────────────────────────────────
    run_eda(df)

    # ── MODULE 4: FEATURE ENGINEERING & SCALING ─────────────────
    scaled_data, scaler, feature_names = run_feature_engineering(
        df, FEATURES
    )

    # ── MODULE 5: SEQUENCE GENERATION ───────────────────────────
    split_data = run_sequence_generation(
        scaled_data, SEQ_LENGTH, TEST_RATIO
    )

    # ── MODULE 6 & 7: BUILD, TRAIN & EVALUATE ──────────────────
    eval_results = train_and_evaluate_all(
        split_data, scaler, EPOCHS, BATCH_SIZE
    )

    # ── MODULE 8: VISUALIZATION ─────────────────────────────────
    run_visualization(
        eval_results["results_df"],
        eval_results["trained_models"],
        HORIZONS
    )

    # ── MODULE 9: SUMMARY ───────────────────────────────────────
    print_summary(eval_results["results_df"])


# ─── Entry Point ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
