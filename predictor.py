import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model


# -----------------------------------------------------------------------------
# CSV/data helpers
# -----------------------------------------------------------------------------

def read_csv_flexible(file_obj):
    """
    Read CSV flexibly for a Streamlit uploaded file or a file path.
    Supports comma and semicolon separators.
    """
    try:
        return pd.read_csv(file_obj, sep=None, engine="python")
    except Exception:
        pass

    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        return pd.read_csv(file_obj, sep=";", decimal=",")
    except Exception:
        pass

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    return pd.read_csv(file_obj, sep=",", decimal=".")


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convert a price column to numeric.
    Supports: 89.81, 89,81, 1,234.56
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s = s.astype(str).str.strip()

    both = s.str.contains(",", regex=False) & s.str.contains(".", regex=False)
    s = s.mask(both, s[both].str.replace(",", "", regex=False))

    only_comma = s.str.contains(",", regex=False) & ~s.str.contains(".", regex=False)
    s = s.mask(only_comma, s[only_comma].str.replace(",", ".", regex=False))

    return pd.to_numeric(s, errors="coerce")


def load_price_series_from_file(
    file_obj,
    date_col: str = "Date",
    price_col: str = "Price",
    dayfirst: bool = True,
) -> pd.Series:
    df = read_csv_flexible(file_obj)
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    if date_col not in df.columns:
        raise ValueError(f"Cannot find date column '{date_col}'. Available columns: {list(df.columns)}")
    if price_col not in df.columns:
        raise ValueError(f"Cannot find price column '{price_col}'. Available columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce")
    df[price_col] = clean_numeric_series(df[price_col])

    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last")

    series = df.set_index(date_col)[price_col].astype(float)
    series.name = price_col
    series.index.name = "date"

    return series


# -----------------------------------------------------------------------------
# Legacy LSTM + SVR loader. This preserves compatibility with the old app bundle.
# -----------------------------------------------------------------------------

def rebuild_lstm_model_from_metadata(meta):
    """
    Rebuild the legacy LSTM architecture from metadata.
    This avoids deserializing the old .keras config directly.
    """
    window_size = int(meta["window_size"])
    lstm_units = int(meta.get("lstm_units", 10))
    dropout = float(meta.get("dropout", 0.05))

    model = Sequential([
        Input(shape=(window_size, 1), name="price_window_input"),
        LSTM(lstm_units, dropout=dropout, name="lstm_feature"),
        Dense(1, name="price_output"),
    ])

    _ = model(np.zeros((1, window_size, 1), dtype=np.float32))
    return model


def _find_lstm_weight_group(h5_file):
    layers_group = h5_file["layers"]
    for layer_name in layers_group.keys():
        layer_group = layers_group[layer_name]
        if "cell" in layer_group and "vars" in layer_group["cell"]:
            vars_group = layer_group["cell"]["vars"]
            if all(str(i) in vars_group for i in range(3)):
                return vars_group
    raise KeyError("Cannot find LSTM weights in model.weights.h5")


def _find_dense_weight_group(h5_file):
    layers_group = h5_file["layers"]
    for layer_name in layers_group.keys():
        layer_group = layers_group[layer_name]
        if "vars" not in layer_group:
            continue
        vars_group = layer_group["vars"]
        if "0" in vars_group and "1" in vars_group:
            w_shape = tuple(vars_group["0"].shape)
            b_shape = tuple(vars_group["1"].shape)
            if len(w_shape) == 2 and w_shape[-1] == 1 and b_shape == (1,):
                return vars_group
    raise KeyError("Cannot find Dense weights in model.weights.h5")


def load_lstm_model_from_keras_weights(lstm_path, meta):
    model = rebuild_lstm_model_from_metadata(meta)

    with zipfile.ZipFile(lstm_path, "r") as z:
        if "model.weights.h5" not in z.namelist():
            raise FileNotFoundError("Cannot find model.weights.h5 in the .keras file")
        weights_bytes = z.read("model.weights.h5")

    with h5py.File(io.BytesIO(weights_bytes), "r") as h5:
        lstm_vars = _find_lstm_weight_group(h5)
        dense_vars = _find_dense_weight_group(h5)
        lstm_weights = [np.array(lstm_vars["0"]), np.array(lstm_vars["1"]), np.array(lstm_vars["2"])]
        dense_weights = [np.array(dense_vars["0"]), np.array(dense_vars["1"])]

    model.get_layer("lstm_feature").set_weights(lstm_weights)
    model.get_layer("price_output").set_weights(dense_weights)
    return model


def build_feature_extractor_safely(lstm, window_size: int) -> Model:
    feature_input = Input(shape=(window_size, 1), name="feature_input")
    x = feature_input
    for layer in lstm.layers:
        x = layer(x)
        if layer.name == "lstm_feature":
            break
    return Model(inputs=feature_input, outputs=x, name="lstm_feature_extractor")


def _load_legacy_lstm_svr_pipeline(model_dir: Path):
    lstm_path = model_dir / "lstm_predictor_model.keras"
    svr_path = model_dir / "svr_regressor.pkl"
    scaler_path = model_dir / "scaler.pkl"
    metadata_path = model_dir / "metadata.json"

    missing = [str(path.name) for path in [lstm_path, svr_path, scaler_path, metadata_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))

    svr = joblib.load(svr_path)
    scaler = joblib.load(scaler_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    lstm = load_lstm_model_from_keras_weights(lstm_path, meta)
    window_size = int(meta["window_size"])
    feature_model = build_feature_extractor_safely(lstm, window_size)

    return lstm, feature_model, svr, scaler, meta


# -----------------------------------------------------------------------------
# New BO + SLSQP weighted ensemble loader.
# It is intentionally exposed through load_lstm_svr_pipeline() so app.py can stay
# unchanged. In the Streamlit app, the variable named "svr" will hold this bundle.
# -----------------------------------------------------------------------------

def _is_ensemble_metadata(meta: Dict[str, Any]) -> bool:
    return meta.get("model_type") == "weighted_ensemble_bo_slsqp"


def _load_ensemble_pipeline(model_dir: Path):
    metadata_path = model_dir / "metadata.json"
    scaler_path = model_dir / "scaler.pkl"
    weights_path = model_dir / "ensemble_weights.npy"

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_names = list(meta.get("model_names", ["cnn", "lstm", "bilstm", "gru", "bigru", "dffnn"]))
    required = [scaler_path, weights_path] + [model_dir / f"{name}.keras" for name in model_names]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ensemble model files: " + ", ".join(missing))

    scaler = joblib.load(scaler_path)
    weights = np.load(weights_path).astype(float)
    if weights.ndim != 1 or len(weights) != len(model_names):
        raise ValueError(f"Invalid ensemble_weights.npy shape: {weights.shape}. Expected ({len(model_names)},)")

    models: Dict[str, Model] = {}
    for name in model_names:
        models[name] = load_model(model_dir / f"{name}.keras", compile=False)

    bundle = {
        "_pipeline_type": "weighted_ensemble_bo_slsqp",
        "models": models,
        "weights": weights,
        "model_names": model_names,
    }

    # Return 5 values to match the old app.py unpacking:
    # lstm, feature_model, svr, scaler, meta
    # Here svr is replaced by the ensemble bundle, while feature_model is unused.
    return None, None, bundle, scaler, meta


def load_lstm_svr_pipeline(model_dir):
    """
    Backward-compatible loader.
    - If metadata.json says model_type = weighted_ensemble_bo_slsqp, load the new ensemble.
    - Otherwise load the legacy LSTM + SVR bundle.
    """
    model_dir = Path(model_dir)
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Missing metadata.json in model directory")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if _is_ensemble_metadata(meta):
        return _load_ensemble_pipeline(model_dir)

    return _load_legacy_lstm_svr_pipeline(model_dir)


# -----------------------------------------------------------------------------
# Prediction functions. Names/signatures are kept unchanged for app.py.
# -----------------------------------------------------------------------------

def _is_ensemble_bundle(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("_pipeline_type") == "weighted_ensemble_bo_slsqp"


def _predict_ensemble_one_step(history_values, ensemble_bundle, scaler, window_size: int) -> float:
    if len(history_values) < window_size:
        raise ValueError(f"Need at least {window_size} observations for forecasting.")

    recent_values = np.asarray(history_values, dtype=float).reshape(-1, 1)[-window_size:]
    recent_scaled = scaler.transform(recent_values)
    x_seq = recent_scaled.reshape(1, window_size, 1).astype(np.float32)
    x_flat = recent_scaled.reshape(1, window_size).astype(np.float32)

    preds_scaled = []
    for name in ensemble_bundle["model_names"]:
        model = ensemble_bundle["models"][name]
        x_in = x_flat if name == "dffnn" else x_seq
        pred = model.predict(x_in, verbose=0)
        preds_scaled.append(float(np.asarray(pred).reshape(-1)[0]))

    weights = np.asarray(ensemble_bundle["weights"], dtype=float)
    pred_scaled = float(np.dot(weights, np.asarray(preds_scaled, dtype=float)))
    pred_price = scaler.inverse_transform(np.array([[pred_scaled]], dtype=float))[0, 0]
    return float(pred_price)


def forecast_next_step_from_series(
    price_series_new: pd.Series,
    feature_model,
    svr,
    scaler,
    window_size: int,
) -> float:
    if len(price_series_new) < window_size:
        raise ValueError(f"Need at least {window_size} observations for forecasting.")

    if _is_ensemble_bundle(svr):
        return _predict_ensemble_one_step(price_series_new.values.astype(float), svr, scaler, window_size)

    recent_values = price_series_new.values.reshape(-1, 1)[-window_size:]
    recent_scaled = scaler.transform(recent_values)
    x_input = recent_scaled.reshape(1, window_size, 1)
    feature = feature_model.predict(x_input, verbose=0)
    pred_scaled = svr.predict(feature).reshape(-1, 1)
    pred_price = scaler.inverse_transform(pred_scaled)[0, 0]
    return float(pred_price)


def forecast_recursive_from_series(
    price_series_new: pd.Series,
    feature_model,
    svr,
    scaler,
    window_size: int,
    steps: int = 30,
    use_business_days: bool = True,
) -> pd.DataFrame:
    if len(price_series_new) < window_size:
        raise ValueError(f"Need at least {window_size} observations for forecasting.")

    history = list(price_series_new.values.astype(float))
    predictions = []

    for _ in range(int(steps)):
        if _is_ensemble_bundle(svr):
            pred_price = _predict_ensemble_one_step(history, svr, scaler, window_size)
        else:
            recent_values = np.array(history[-window_size:]).reshape(-1, 1)
            recent_scaled = scaler.transform(recent_values)
            x_input = recent_scaled.reshape(1, window_size, 1)
            feature = feature_model.predict(x_input, verbose=0)
            pred_scaled = svr.predict(feature).reshape(-1, 1)
            pred_price = scaler.inverse_transform(pred_scaled)[0, 0]

        predictions.append(float(pred_price))
        history.append(float(pred_price))

    last_date = price_series_new.index[-1]

    if use_business_days:
        future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=int(steps))
    else:
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=int(steps), freq="D")

    return pd.DataFrame({"date": future_dates, "forecast": predictions})
