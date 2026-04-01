# 🛡️ Credit Card Fraud Detection — Streamlit App

Ứng dụng học máy phát hiện gian lận thẻ tín dụng, xây dựng bằng **Logistic Regression** và **XGBoost**.  
Đồ án tốt nghiệp · Võ Nguyễn Huyền Vi · HUFI 2024

---

## 📁 Cấu trúc thư mục

Trước khi chạy, hãy đảm bảo repo có đúng cấu trúc sau:

```
📦 project/
├── app.py
├── requirements.txt
├── creditcard.csv                        ← Dataset gốc từ Kaggle
│
├── models/
│   ├── logistic_regression (1).pkl
│   ├── xgboost_model (1).pkl
│   └── scaler (1).pkl
│
├── images/                               ← Ảnh EDA (Trang 1)
│   ├── 1_fraud_time.png
│   ├── 2_all_time.png
│   ├── 3_correlation_matrix.png
│   └── 4_class_distribution.png
│
└── hehehehehehe/                         ← Ảnh đánh giá mô hình (Trang 3)
    ├── 05_confusion_matrix.png
    ├── 06_roc_curve.png
    ├── 07_precision_recall_curve.png
    ├── 08_metrics_comparison.png
    ├── 09_xgb_feature_importance.png
    ├── 10_lr_coefficients.png
    ├── 11_learning_curve.png
    ├── 12_xgb_training_history.png
    ├── 13_probability_distribution.png
    ├── 14_threshold_analysis.png
    ├── 15_error_analysis.png
    └── 16_classification_report_heatmap.png
```

> **Lưu ý:** Dataset `creditcard.csv` có thể tải tại [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
> File này **không được commit lên GitHub** vì dung lượng lớn (~150MB). Hãy thêm vào `.gitignore`.

---

## 🚀 Chạy local

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Chạy app
streamlit run app.py
```

---

## ☁️ Deploy lên Streamlit Community Cloud

1. Push toàn bộ code lên **GitHub** (bao gồm `app.py`, `requirements.txt`, các folder `models/`, `images/`, `hehehehehehe/`).
2. Truy cập [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Chọn repo, branch `main`, file `app.py` → **Deploy**.

> ⚠️ **Quan trọng:** Streamlit Community Cloud có giới hạn **1GB RAM** và không cho phép upload file lớn trực tiếp.  
> Nếu `creditcard.csv` quá lớn, hãy dùng [Streamlit Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) hoặc lưu dataset trên Google Drive / Hugging Face Datasets rồi tải về khi khởi động.

---

## 📊 Tính năng

| Trang | Nội dung |
|-------|----------|
| 📊 Giới thiệu & Dữ liệu | Tổng quan bài toán, mẫu dữ liệu ngẫu nhiên, 4 biểu đồ EDA |
| 🤖 Triển khai mô hình | Nhập tay hoặc random giao dịch, dự đoán bằng LR / XGBoost, hiển thị xác suất |
| 📈 Huấn luyện & Đánh giá | 12 biểu đồ phân tích: confusion matrix, ROC, PR curve, feature importance, learning curve... |

---

## 🛠️ Tech stack

- **Python 3.10+**
- [Streamlit](https://streamlit.io/) — giao diện web
- [XGBoost](https://xgboost.readthedocs.io/) — mô hình gradient boosting
- [scikit-learn](https://scikit-learn.org/) — Logistic Regression, scaler, metrics
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — xử lý dữ liệu
- [Matplotlib](https://matplotlib.org/) — biểu đồ inline

---

## 📝 Ghi chú

- Mô hình đã được huấn luyện sẵn và lưu dưới dạng `.pkl` — app chỉ load và inference, không train lại.
- Kỹ thuật **SMOTE** + **StandardScaler** được áp dụng trong quá trình huấn luyện.
- Ngưỡng phân loại mặc định là **0.5**; có thể điều chỉnh theo phân tích ở Trang 3.
