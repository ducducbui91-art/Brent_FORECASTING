# Dated Brent Forecast Streamlit App

App dự báo giá Dated Brent bằng mô hình đã train:

```text
LSTM feature extractor + SVR regressor
```

## 1. Cấu trúc thư mục

```text
dated_brent_streamlit_app/
├── app.py
├── predictor.py
├── requirements.txt
├── README.md
├── .gitignore
└── trained_lstm_svr_model/
    ├── lstm_predictor_model.keras
    ├── svr_regressor.pkl
    ├── scaler.pkl
    └── metadata.json
```

## 2. Copy model đã train

Sau khi chạy notebook train thành công, copy các file sau vào thư mục `trained_lstm_svr_model/`:

```text
lstm_predictor_model.keras
svr_regressor.pkl
scaler.pkl
metadata.json
```

## 3. Cài môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trên Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Chạy app local

```bash
streamlit run app.py
```

## 5. Format file CSV upload

Ví dụ:

```text
Date;Dated_Brent
01/01/2024;77,72
02/01/2024;78,25
```

Hoặc:

```text
Date,Dated_Brent
2024-01-01,77.72
2024-01-02,78.25
```

## 6. Đưa lên GitHub

Nên đưa lên GitHub:

```text
app.py
predictor.py
requirements.txt
README.md
.gitignore
```

Nếu model không quá lớn và repo private, có thể đưa thêm:

```text
trained_lstm_svr_model/lstm_predictor_model.keras
trained_lstm_svr_model/svr_regressor.pkl
trained_lstm_svr_model/scaler.pkl
trained_lstm_svr_model/metadata.json
```

Không nên đưa dữ liệu nội bộ hoặc dữ liệu nhạy cảm lên repo public.
