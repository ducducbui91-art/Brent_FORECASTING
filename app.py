import io
from pathlib import Path

import pandas as pd
import streamlit as st

from predictor import (
    load_price_series_from_file,
    load_lstm_svr_pipeline,
    forecast_next_step_from_series,
    forecast_recursive_from_series,
)


st.set_page_config(
    page_title="Dated Brent Forecast App",
    page_icon="🛢️",
    layout="wide",
)


DEFAULT_MODEL_DIR = Path("trained_lstm_svr_model")


st.title("Dated Brent Forecast App")
st.caption("LSTM feature extractor + SVR regressor")

st.markdown(
    """
App này dùng bộ model đã train sẵn:

Bạn upload file CSV giá Brent mới nhất, app sẽ dự báo giá Dated Brent cho các bước tiếp theo.
"""
)


with st.sidebar:
    st.header("Cấu hình")

    model_dir = st.text_input(
        "Thư mục model",
        value=str(DEFAULT_MODEL_DIR),
        help="Thư mục chứa lstm_predictor_model.keras, svr_regressor.pkl, scaler.pkl, metadata.json",
    )

    forecast_steps = st.number_input(
        "Số bước muốn dự báo",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
    )

    use_business_days = st.checkbox(
        "Tạo ngày dự báo theo ngày làm việc",
        value=True,
    )

    st.divider()

    st.markdown(
        """
**Format CSV khuyến nghị**

```text
Date;Price
01/01/2024;77,72
02/01/2024;78,25
```

Hoặc:

```text
Date,Price
2024-01-01,77.72
2024-01-02,78.25
```
"""
    )


uploaded_file = st.file_uploader(
    "Upload file CSV dữ liệu Dated Brent mới nhất",
    type=["csv"],
)


if uploaded_file is None:
    st.info("Hãy upload file CSV để chạy dự báo.")
    st.stop()


try:
    lstm, feature_model, svr, scaler, meta = load_lstm_svr_pipeline(model_dir)
except Exception as exc:
    st.error("Không load được model. Kiểm tra lại thư mục model và các file model đã lưu.")
    st.exception(exc)
    st.stop()


try:
    price_series = load_price_series_from_file(
        uploaded_file,
        date_col=meta.get("date_col", "Date"),
        price_col=meta.get("price_col", "Dated_Brent"),
    )
except Exception as exc:
    st.error("Không đọc được file CSV. Kiểm tra tên cột Date và Dated_Brent.")
    st.exception(exc)
    st.stop()


window_size = int(meta["window_size"])

if len(price_series) < window_size:
    st.error(f"Dữ liệu cần ít nhất {window_size} quan sát để dự báo. File hiện có {len(price_series)} quan sát.")
    st.stop()


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Số quan sát", f"{len(price_series):,}")

with col2:
    st.metric("Ngày đầu", str(price_series.index.min().date()))

with col3:
    st.metric("Ngày cuối", str(price_series.index.max().date()))

with col4:
    st.metric("Giá mới nhất", f"{price_series.iloc[-1]:,.2f}")


with st.expander("Xem metadata model", expanded=False):
    st.json(meta)


st.subheader("Dữ liệu thực tế gần đây")

recent_df = price_series.tail(100).reset_index()
recent_df.columns = ["date", "actual_price"]

st.line_chart(
    recent_df.set_index("date")["actual_price"],
    height=320,
)


if st.button("Chạy dự báo", type="primary"):
    try:
        next_price = forecast_next_step_from_series(
            price_series_new=price_series,
            feature_model=feature_model,
            svr=svr,
            scaler=scaler,
            window_size=window_size,
        )

        forecast_df = forecast_recursive_from_series(
            price_series_new=price_series,
            feature_model=feature_model,
            svr=svr,
            scaler=scaler,
            window_size=window_size,
            steps=int(forecast_steps),
            use_business_days=use_business_days,
        )

    except Exception as exc:
        st.error("Có lỗi khi chạy dự báo.")
        st.exception(exc)
        st.stop()

    st.success("Đã chạy dự báo xong.")

    st.subheader("Kết quả nhanh")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Giá mới nhất", f"{price_series.iloc[-1]:,.2f}")

    with c2:
        st.metric("Dự báo bước tiếp theo", f"{next_price:,.2f}")

    with c3:
        change = next_price - price_series.iloc[-1]
        st.metric("Chênh lệch bước tiếp theo", f"{change:,.2f}")

    st.subheader("Bảng dự báo")
    st.dataframe(forecast_df, use_container_width=True)

    st.subheader("Biểu đồ thực tế + dự báo")

    actual_plot = price_series.tail(120).reset_index()
    actual_plot.columns = ["date", "actual"]

    forecast_plot = forecast_df.copy()
    forecast_plot = forecast_plot.rename(columns={"forecast": "forecast"})

    plot_df = pd.merge(
        actual_plot,
        forecast_plot,
        on="date",
        how="outer",
    ).sort_values("date")

    st.line_chart(
        plot_df.set_index("date")[["actual", "forecast"]],
        height=420,
    )

    csv_bytes = forecast_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    st.download_button(
        label="Tải file forecast CSV",
        data=csv_bytes,
        file_name="dated_brent_forecast.csv",
        mime="text/csv",
    )

    st.warning(
        "Lưu ý: dự báo nhiều bước là dự báo cuốn chiếu, càng dự báo xa thì sai số tích lũy càng lớn."
    )
