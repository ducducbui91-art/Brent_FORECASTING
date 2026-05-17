import io
import json
import zipfile
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Dense


def read_csv_flexible(file_obj):
    """
    Đọc CSV linh hoạt cho Streamlit uploaded file hoặc đường dẫn file.
    Hỗ trợ cả dấu ; và dấu ,.
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
    Chuyển cột giá về dạng số.
    Hỗ trợ:
    - 89.81
    - 89,81
    - 1,234.56
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
    # Xử lý tên cột, bao gồm cả ký tự BOM ẩn ở đầu file CSV
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    if date_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột ngày '{date_col}'. Các cột hiện có: {list(df.columns)}")

    if price_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột giá '{price_col}'. Các cột hiện có: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce")
    df[price_col] = clean_numeric_series(df[price_col])

    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last")

    series = df.set_index(date_col)[price_col].astype(float)
    series.name = price_col
    series.index.name = "date"

    return series


def rebuild_lstm_model_from_metadata(meta):
    """
    Dựng lại kiến trúc LSTM từ metadata.
    Không deserialize config .keras, nên tránh lỗi:
    Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
    """
    window_size = int(meta["window_size"])
    lstm_units = int(meta.get("lstm_units", 10))
    dropout = float(meta.get("dropout", 0.05))

    model = Sequential([
        Input(shape=(window_size, 1), name="price_window_input"),
        LSTM(lstm_units, dropout=dropout, name="lstm_feature"),
        Dense(1, name="price_output"),
    ])

    # Build model bằng dummy input
    _ = model(np.zeros((1, window_size, 1), dtype=np.float32))

    return model


def _find_lstm_weight_group(h5_file):
    """
    Tìm group chứa weights của LSTM trong model.weights.h5.
    Với file hiện tại thường là: layers/lstm/cell/vars
    """
    layers_group = h5_file["layers"]

    for layer_name in layers_group.keys():
        layer_group = layers_group[layer_name]
        if "cell" in layer_group and "vars" in layer_group["cell"]:
            vars_group = layer_group["cell"]["vars"]
            if all(str(i) in vars_group for i in range(3)):
                return vars_group

    raise KeyError("Không tìm thấy group weights của LSTM trong model.weights.h5")


def _find_dense_weight_group(h5_file):
    """
    Tìm group chứa weights của Dense trong model.weights.h5.
    Với file hiện tại thường là: layers/dense/vars
    """
    layers_group = h5_file["layers"]

    for layer_name in layers_group.keys():
        layer_group = layers_group[layer_name]
        if "vars" not in layer_group:
            continue

        vars_group = layer_group["vars"]
        if "0" in vars_group and "1" in vars_group:
            w_shape = tuple(vars_group["0"].shape)
            b_shape = tuple(vars_group["1"].shape)

            # Dense cuối có dạng kernel=(units, 1), bias=(1,)
            if len(w_shape) == 2 and w_shape[-1] == 1 and b_shape == (1,):
                return vars_group

    raise KeyError("Không tìm thấy group weights của Dense trong model.weights.h5")


def load_lstm_model_from_keras_weights(lstm_path, meta):
    """
    Load LSTM model bằng cách:
    1. Dựng lại architecture từ metadata
    2. Mở file .keras như zip
    3. Đọc model.weights.h5
    4. Set weights thủ công vào LSTM và Dense

    Cách này không gọi load_model(), nên không bị lỗi deserialize Dense quantization_config.
    """
    model = rebuild_lstm_model_from_metadata(meta)

    with zipfile.ZipFile(lstm_path, "r") as z:
        if "model.weights.h5" not in z.namelist():
            raise FileNotFoundError("Không tìm thấy model.weights.h5 trong file .keras")

        weights_bytes = z.read("model.weights.h5")

    with h5py.File(io.BytesIO(weights_bytes), "r") as h5:
        lstm_vars = _find_lstm_weight_group(h5)
        dense_vars = _find_dense_weight_group(h5)

        lstm_weights = [
            np.array(lstm_vars["0"]),
            np.array(lstm_vars["1"]),
            np.array(lstm_vars["2"]),
        ]

        dense_weights = [
            np.array(dense_vars["0"]),
            np.array(dense_vars["1"]),
        ]

    model.get_layer("lstm_feature").set_weights(lstm_weights)
    model.get_layer("price_output").set_weights(dense_weights)

    return model


def build_feature_extractor_safely(lstm, window_size: int) -> Model:
    """
    Tạo feature extractor an toàn, không phụ thuộc vào lstm.input.
    """
    feature_input = Input(shape=(window_size, 1), name="feature_input")
    x = feature_input

    for layer in lstm.layers:
        x = layer(x)
        if layer.name == "lstm_feature":
            break

    return Model(
        inputs=feature_input,
        outputs=x,
        name="lstm_feature_extractor",
    )


def load_lstm_svr_pipeline(model_dir):
    model_dir = Path(model_dir)

    lstm_path = model_dir / "lstm_predictor_model.keras"
    svr_path = model_dir / "svr_regressor.pkl"
    scaler_path = model_dir / "scaler.pkl"
    metadata_path = model_dir / "metadata.json"

    missing = [
        str(path.name)
        for path in [lstm_path, svr_path, scaler_path, metadata_path]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Thiếu file model: " + ", ".join(missing)
        )

    svr = joblib.load(svr_path)
    scaler = joblib.load(scaler_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Quan trọng: không dùng load_model(lstm_path)
    lstm = load_lstm_model_from_keras_weights(lstm_path, meta)

    window_size = int(meta["window_size"])
    feature_model = build_feature_extractor_safely(lstm, window_size)

    return lstm, feature_model, svr, scaler, meta


def forecast_next_step_from_series(
    price_series_new: pd.Series,
    feature_model,
    svr,
    scaler,
    window_size: int,
) -> float:
    if len(price_series_new) < window_size:
        raise ValueError(f"Cần ít nhất {window_size} quan sát để dự báo.")

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
        raise ValueError(f"Cần ít nhất {window_size} quan sát để dự báo.")

    history = list(price_series_new.values.astype(float))
    predictions = []

    for _ in range(int(steps)):
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
        future_dates = pd.bdate_range(
            start=last_date + pd.offsets.BDay(1),
            periods=int(steps),
        )
    else:
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=int(steps),
            freq="D",
        )

    return pd.DataFrame({
        "date": future_dates,
        "forecast": predictions,
    })
