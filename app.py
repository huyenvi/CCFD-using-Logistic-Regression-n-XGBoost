# -*- coding: utf-8 -*-
# Credit Card Fraud Detection — Streamlit App
# Trang 1: Giới thiệu & Khám phá dữ liệu
# Trang 2: Triển khai mô hình & Dự đoán
# Trang 3: Đánh giá & Hiệu năng

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PAGE CONFIG — phải là lệnh Streamlit đầu tiên
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection | HUSC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1180px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #f1f5f9 !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 70%, #2563eb 100%);
    border-radius: 16px; padding: 2.2rem 2.6rem; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(30,64,175,.18);
}
.hero::after {
    content: ''; position: absolute; top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(255,255,255,.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block; background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.25); border-radius: 20px;
    padding: .25rem .9rem; font-size: .75rem; color: #bfdbfe;
    margin-bottom: .8rem; letter-spacing: .04em;
}
.hero-title { font-size: 1.55rem; font-weight: 800; color: #fff; line-height: 1.45; margin-bottom: .35rem; }
.hero-title span { color: #93c5fd; }
.hero-sub { font-size: .88rem; color: rgba(255,255,255,.65); margin-bottom: 1.1rem; }
.hero-divider { border: none; border-top: 1px solid rgba(255,255,255,.15); margin: 1rem 0; }
.hero-meta { display: flex; gap: 2.5rem; flex-wrap: wrap; font-size: .85rem; color: rgba(255,255,255,.75); }
.hero-meta b { color: #fff; }

/* Section header */
.section-header {
    font-size: 1rem; font-weight: 700; color: #1e293b;
    border-left: 4px solid #2563eb; padding-left: .75rem; margin: 2rem 0 1rem;
}

/* Value box */
.value-box {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.2rem 1.4rem; box-shadow: 0 2px 8px rgba(0,0,0,.04);
    font-size: .9rem; line-height: 1.75; color: #334155;
}
.value-box ul { margin: .5rem 0 0 .3rem; padding-left: 1.1rem; }
.value-box li { margin-bottom: .3rem; }

/* KPI cards */
.kpi-row { display: flex; gap: .85rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 120px; background: #fff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.1rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,.04); border-top: 3px solid #2563eb;
}
.kpi-card.green { border-top-color: #10b981; }
.kpi-card.red   { border-top-color: #ef4444; }
.kpi-card.amber { border-top-color: #f59e0b; }
.kpi-label { font-size: .68rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { font-size: 1.45rem; font-weight: 800; color: #1e293b; margin-top: .2rem; }
.kpi-sub   { font-size: .7rem; color: #94a3b8; margin-top: .15rem; }

/* Chart label */
.chart-label {
    font-size: .8rem; font-weight: 600; color: #475569;
    background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 6px; padding: .25rem .75rem;
    display: inline-block; margin-bottom: .5rem;
}

/* Input group title */
.input-group-title {
    font-size: .82rem; font-weight: 700; color: #1e3a5f;
    background: linear-gradient(90deg, #eff6ff, #f8fafc);
    border-left: 3px solid #3b82f6;
    padding: .4rem .85rem; border-radius: 0 6px 6px 0;
    margin-bottom: .5rem; margin-top: .9rem; letter-spacing: .03em;
}

/* Pipeline wrap */
.pipe-wrap {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 1.2rem 1.2rem .8rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.05); margin-bottom: .4rem;
}

/* Step detail */
.step-detail {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: .8rem 1rem; font-size: .79rem; color: #475569;
    line-height: 1.7; height: 100%;
}
.step-detail b { color: #1e3a5f; }

/* Result outer card */
.result-outer {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 1.6rem 1.5rem 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,.06);
}
.model-badge {
    display: inline-block; background: #1e3a5f; color: #fff;
    font-size: .72rem; font-weight: 700; border-radius: 20px;
    padding: .22rem .9rem; margin-bottom: 1rem; letter-spacing: .05em;
}
.result-card {
    border-radius: 12px; padding: 1.4rem 1.2rem 1.3rem; text-align: center;
}
.result-card.legit { background: linear-gradient(135deg,#ecfdf5,#d1fae5); border: 2px solid #10b981; }
.result-card.fraud { background: linear-gradient(135deg,#fff1f2,#ffe4e6); border: 2px solid #ef4444; }
.result-icon  { font-size: 3rem; line-height: 1; }
.result-label { font-size: 1.2rem; font-weight: 800; margin-top: .5rem; }
.result-label.legit { color: #065f46; }
.result-label.fraud { color: #991b1b; }
.result-prob  { font-size: .83rem; color: #475569; margin-top: .45rem; line-height: 1.5; }

/* Probability bar */
.prob-bar-wrap { margin-top: 1rem; }
.prob-bar-label {
    display: flex; justify-content: space-between;
    font-size: .76rem; color: #64748b; margin-bottom: .3rem;
}
.prob-bar-bg { background: #f1f5f9; border-radius: 99px; height: 11px; overflow: hidden; }
.prob-bar-fill-legit { background: linear-gradient(90deg,#10b981,#34d399); height: 100%; border-radius: 99px; }
.prob-bar-fill-fraud { background: linear-gradient(90deg,#ef4444,#f87171); height: 100%; border-radius: 99px; }

/* Comment box (trang 3) */
.cmt-good    { background:#f0fdf4; border-left:4px solid #10b981; border-radius:0 8px 8px 0; padding:.75rem 1.1rem; font-size:.83rem; color:#065f46; line-height:1.7; margin-top:.5rem; }
.cmt-warn    { background:#fffbeb; border-left:4px solid #f59e0b; border-radius:0 8px 8px 0; padding:.75rem 1.1rem; font-size:.83rem; color:#78350f; line-height:1.7; margin-top:.5rem; }
.cmt-improve { background:#eff6ff; border-left:4px solid #3b82f6; border-radius:0 8px 8px 0; padding:.75rem 1.1rem; font-size:.83rem; color:#1e3a5f; line-height:1.7; margin-top:.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:.8rem 0 1.2rem;'>
        <div style='font-size:2.2rem;'>🛡️</div>
        <div style='font-weight:800; font-size:.95rem; color:#f1f5f9; margin-top:.3rem;'>Fraud Detector</div>
        <div style='font-size:.72rem; color:#64748b; margin-top:.15rem;'>Credit Card · ML App</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["📊  Giới thiệu & Dữ liệu",
         "🤖  Triển khai mô hình",
         "📈  Huấn luyện & Đánh giá"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#334155; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding-top:2.5rem; font-size:.7rem; color:#475569; text-align:center;'>
        HUSC · Khoa CNTT · 2026
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN — Sửa tại đây
# ─────────────────────────────────────────────

# ================================================================
#  BƯỚC 1: Điền đường dẫn tuyệt đối đến file creditcard.csv
#  Ví dụ Windows : r"C:\Users\TenBan\Desktop\creditcard.csv"
# ================================================================
CSV_PATH = r"data/creditcard.csv"

# ================================================================
#  BƯỚC 2: Điền đường dẫn tuyệt đối đến 4 ảnh EDA
# ================================================================
IMG_CLASS_DIST  = r"images/4_class_distribution.png"
IMG_AMOUNT_DIST = r"images/2_all_time.png"
IMG_CORR        = r"images/3_correlation_matrix.png"
IMG_TIME        = r"images/1_fraud_time.png"

# ================================================================
#  BƯỚC 3: Điền đường dẫn đến file model đã huấn luyện (.pkl)
# ================================================================
MODEL_LR_PATH  = r"models/logistic_regression (1).pkl"
MODEL_XGB_PATH = r"models/xgboost_model (1).pkl"
SCALER_PATH    = r"models/scaler (1).pkl"

# ================================================================
#  BƯỚC 4: Điền đường dẫn đến 12 ảnh đánh giá (trang 3)
#  Tất cả nằm trong folder hehehehehehe sau khi chạy Kaggle
# ================================================================
IMG_05 = r"images/05_confusion_matrix.png"
IMG_06 = r"images/06_roc_curve.png"
IMG_07 = r"images/07_precision_recall_curve.png"
IMG_08 = r"images/08_metrics_comparison.png"
IMG_09 = r"images/09_xgb_feature_importance.png"
IMG_10 = r"images/10_lr_coefficients.png"
IMG_11 = r"images/11_learning_curve.png"
IMG_12 = r"images/12_xgb_training_history.png"
IMG_13 = r"images/13_probability_distribution.png"
IMG_14 = r"images/14_threshold_analysis.png"
IMG_15 = r"images/15_error_analysis.png"
IMG_16 = r"images/16_classification_report_heatmap.png"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return None

@st.cache_resource(show_spinner=False)
def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — GIỚI THIỆU & KHÁM PHÁ DỮ LIỆU
# ══════════════════════════════════════════════════════════════════
if page == "📊  Giới thiệu & Dữ liệu":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🎓 Đánh Giá Cuối Học Phần</div>
        <div class="hero-title">
            Phân tích dữ liệu giao dịch thẻ tín dụng <br>
            <span>bằng thuật toán Logistic Regression và XGBoost </span>
        </div>
        <div class="hero-sub">: nhằm nhận diện và cảnh báo các giao dịch gian lận</div>
        <hr class="hero-divider">
        <div class="hero-meta">
            <span>👤 <b>Họ và tên:</b> Võ Nguyễn Huyền Vi</span>
            <span>🎓 <b>Mã Sinh Viên:</b> 21T1020828</span>
            <span>🏫 <b>Trường:</b> Đại Học Khoa Học </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">💡 Giá trị thực tiễn của bài toán</div>',
                unsafe_allow_html=True)
    col_desc, col_kpi = st.columns([3, 2], gap="large")

    with col_desc:
        st.markdown("""
        <div class="value-box">
            Gian lận thẻ tín dụng là một trong những vấn đề tài chính nghiêm trọng nhất hiện nay,
            gây thiệt hại <b>hàng chục tỷ USD</b> mỗi năm trên toàn cầu. Với sự bùng nổ của
            thanh toán số, các tổ chức tài chính ngày càng cần hệ thống phát hiện gian lận
            <b>tự động, nhanh chóng và chính xác</b>.
            <br><br>
            Thách thức lớn nhất là dữ liệu <b>cực kỳ mất cân bằng</b> — giao dịch gian lận
            chỉ chiếm chưa đến 0.2% tổng số giao dịch. Nghiên cứu xây dựng và so sánh:
            <ul>
                <li><b>Logistic Regression</b> — nền tảng, dễ giải thích, triển khai nhanh</li>
                <li><b>XGBoost</b> — ensemble mạnh mẽ, xử lý tốt dữ liệu phi tuyến</li>
            </ul>
            Kết hợp kỹ thuật <b>SMOTE</b> để cân bằng dữ liệu và <b>StandardScaler</b>
            để chuẩn hoá đặc trưng.
        </div>
        """, unsafe_allow_html=True)

    with col_kpi:
        st.markdown("""
        <div class="kpi-row">
            <div class="kpi-card red">
                <div class="kpi-label">Thiệt hại / năm</div>
                <div class="kpi-value">$32B+</div>
                <div class="kpi-sub">toàn cầu</div>
            </div>
            <div class="kpi-card amber">
                <div class="kpi-label">Tỷ lệ gian lận</div>
                <div class="kpi-value">~0.17%</div>
                <div class="kpi-sub">mất cân bằng cao</div>
            </div>
        </div>
        <div class="kpi-row" style="margin-top:.85rem;">
            <div class="kpi-card green">
                <div class="kpi-label">Mô hình</div>
                <div class="kpi-value">2</div>
                <div class="kpi-sub">LR · XGBoost</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Đặc trưng</div>
                <div class="kpi-value">30</div>
                <div class="kpi-sub">V1–V28, Amount, Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    df = load_data()

    st.markdown('<div class="section-header">📋 Mẫu dữ liệu (10 dòng ngẫu nhiên)</div>',
                unsafe_allow_html=True)

    if df is not None:
        if st.button("🔀 Chọn lại ngẫu nhiên"):
            st.session_state["sample_seed"] = np.random.randint(0, 9999)
        seed = st.session_state.get("sample_seed", 42)
        sample = df.sample(10, random_state=seed)
        show_cols = ["Time"] + [f"V{i}" for i in range(1, 6)] + ["Amount", "Class"]
        display_df = sample[show_cols].copy().round(4)
        display_df["Class"] = display_df["Class"].map({0: "✅ Hợp lệ", 1: "🚨 Gian lận"})
        st.dataframe(display_df, use_container_width=True, height=390)
        st.caption(
            f"Hiển thị 7 / {len(df.columns)} cột · "
            f"Tổng **{len(df):,}** giao dịch · "
            f"Gian lận: **{df['Class'].sum():,}** ({df['Class'].mean()*100:.3f}%)"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Tổng giao dịch", f"{len(df):,}")
        c2.metric("✅ Hợp lệ",         f"{(df['Class']==0).sum():,}")
        c3.metric("🚨 Gian lận",       f"{(df['Class']==1).sum():,}")
        c4.metric("⚖️ Tỷ lệ fraud",    f"{df['Class'].mean()*100:.3f}%")
    else:
        st.warning(
            f"⚠️ Không tìm thấy file CSV tại:\n\n`{CSV_PATH}`\n\n"
            "Hãy điền đường dẫn đúng vào biến `CSV_PATH` trong code."
        )

    st.markdown('<div class="section-header">📈 Biểu đồ khám phá dữ liệu (EDA)</div>',
                unsafe_allow_html=True)

    def show_chart(img_path, icon, label):
        st.markdown(f'<span class="chart-label">{icon} {label}</span>',
                    unsafe_allow_html=True)
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        else:
            st.warning(f"⚠️ Không tìm thấy ảnh tại: `{img_path}`")

    col1, col2 = st.columns(2, gap="medium")
    with col1: show_chart(IMG_CLASS_DIST,  "📊", "Phân phối nhãn (Class)")
    with col2: show_chart(IMG_AMOUNT_DIST, "💰", "Phân bố giá trị giao dịch (Amount)")
    st.markdown("<div style='margin-top:.6rem;'></div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2, gap="medium")
    with col3: show_chart(IMG_CORR, "🔥", "Ma trận tương quan")
    with col4: show_chart(IMG_TIME, "⏱️", "Phân bố theo thời gian (Time)")


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — TRIỂN KHAI MÔ HÌNH
# ══════════════════════════════════════════════════════════════════
elif page == "🤖  Triển khai mô hình":

    model_lr  = load_model(MODEL_LR_PATH)
    model_xgb = load_model(MODEL_XGB_PATH)
    scaler    = load_model(SCALER_PATH)

    # ── Hero ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🤖 Trang 2 · Triển khai mô hình</div>
        <div class="hero-title">
            Dự đoán giao dịch <span>gian lận thẻ tín dụng</span>
        </div>
        <div class="hero-sub">Nhập thông tin giao dịch → Tiền xử lý → Dự đoán bằng LR &amp; XGBoost</div>
        <hr class="hero-divider">
        <div class="hero-meta">
            <span>📥 <b>Input:</b> 30 đặc trưng (Time · V1–V28 · Amount)</span>
            <span>⚙️ <b>Preprocessing:</b> StandardScaler</span>
            <span>🎯 <b>Output:</b> Hợp lệ / Gian lận + Xác suất</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════
    # DỮ LIỆU MẪU SẴN CÓ (lấy thực từ dataset)
    # ════════════════════════════════════════
    PRESETS = {
        "✅ Mẫu 1 — Hợp lệ (giao dịch nhỏ)": {
            "Time": 406.0,    "Amount": 2.69,
            "V1": -2.3122, "V2": 1.9519,  "V3": -1.6098, "V4": 3.9979,
            "V5": -0.5221, "V6": -1.4265, "V7": -2.5374, "V8": 1.3918,
            "V9": -2.7706, "V10": -2.7722,"V11": 3.2020, "V12": -2.8995,
            "V13": -0.5950,"V14": -4.2898,"V15": 0.3898, "V16": -1.1407,
            "V17": -2.8305,"V18": -0.0168,"V19": 0.4164, "V20": 0.3594,
            "V21": 0.0585, "V22": -0.3320,"V23": -0.1634,"V24": -0.1977,
            "V25": -0.0636,"V26": 0.0759, "V27": -0.2261,"V28": -0.1682,
        },
        "✅ Mẫu 2 — Hợp lệ (giao dịch lớn)": {
            "Time": 84692.0, "Amount": 149.62,
            "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,  "V4": 1.3782,
            "V5": -0.3383, "V6": 0.4624,  "V7": 0.2396,  "V8": 0.0987,
            "V9": 0.3638,  "V10": 0.0908, "V11": -0.5516,"V12": -0.6178,
            "V13": -0.9913,"V14": -0.3112,"V15": 1.4681, "V16": -0.4704,
            "V17": 0.2079, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
            "V21": -0.0183,"V22": 0.2778, "V23": -0.1105,"V24": 0.0669,
            "V25": 0.1285, "V26": -0.1891,"V27": 0.1336, "V28": -0.0211,
        },
        "🚨 Mẫu 3 — Gian lận (điển hình)": {
            "Time": 406.0,   "Amount": 1.00,
            "V1": -3.0435, "V2": -3.1572, "V3": 1.0880,  "V4": 2.2886,
            "V5": 1.3597,  "V6": -1.0635, "V7": 0.3253,  "V8": -0.0678,
            "V9": -0.2704, "V10": -0.8382,"V11": -0.4147,"V12": -0.5027,
            "V13": 0.6760, "V14": -1.1196,"V15": 0.1751, "V16": -0.4514,
            "V17": -0.2370,"V18": -0.0381,"V19": 0.8031, "V20": 0.4085,
            "V21": -0.0093,"V22": 0.7989, "V23": -0.1378,"V24": 0.1412,
            "V25": 0.2009, "V26": 0.0752, "V27": 0.0528, "V28": 0.0036,
        },
        "🚨 Mẫu 4 — Gian lận (giá trị cao)": {
            "Time": 40890.0, "Amount": 245.00,
            "V1": -2.3028, "V2": 1.7594,  "V3": -0.3580, "V4": 2.3303,
            "V5": -0.8218, "V6": -1.9878, "V7": 0.1325,  "V8": -0.2267,
            "V9": -0.4592, "V10": -1.9585,"V11": 1.7929, "V12": -1.9117,
            "V13": -0.3803,"V14": -1.2825,"V15": 0.2552, "V16": -0.9517,
            "V17": -1.4637,"V18": -0.2012,"V19": 0.7225, "V20": 0.1099,
            "V21": 0.0232, "V22": 0.3909, "V23": -0.1496,"V24": -0.3088,
            "V25": 0.0275, "V26": -0.1456,"V27": -0.0498,"V28": 0.0613,
        },
        "✏️ Tự nhập giá trị": None,
    }

    # ════════════════════════════════════════
    # PHẦN 1 — CHỌN MẪU + FORM NHẬP LIỆU
    # ════════════════════════════════════════
    st.markdown('<div class="section-header">📥 Nhập thông tin giao dịch</div>',
                unsafe_allow_html=True)

    sel_col, info_col = st.columns([2, 3], gap="medium")
    with sel_col:
        preset_choice = st.selectbox(
            "⚡ Chọn nhanh dữ liệu mẫu",
            options=list(PRESETS.keys()),
            index=0,
            help="Chọn mẫu có sẵn để điền tự động vào tất cả ô, hoặc chọn 'Tự nhập' để nhập tay"
        )
    with info_col:
        if preset_choice != "✏️ Tự nhập giá trị":
            is_fraud_hint = "🚨" in preset_choice
            bg  = "#fef2f2" if is_fraud_hint else "#f0fdf4"
            bdr = "#ef4444" if is_fraud_hint else "#10b981"
            tc  = "#991b1b" if is_fraud_hint else "#065f46"
            lbl = "🚨 GIAN LẬN" if is_fraud_hint else "✅ HỢP LỆ"
            desc = preset_choice.split("—")[1].strip() if "—" in preset_choice else ""
            st.markdown(f"""
            <div style="background:{bg};border:1.5px solid {bdr};border-radius:10px;
                        padding:.65rem 1.1rem;margin-top:.3rem;font-size:.83rem;
                        color:{tc};line-height:1.7;">
                <b>Nhãn thực tế:</b> {lbl} &nbsp;·&nbsp; <b>Mô tả:</b> {desc}<br>
                Các ô bên dưới đã được <b>điền sẵn</b> — nhấn 🚀 để dự đoán ngay.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#fffbeb;border:1.5px solid #f59e0b;border-radius:10px;
                        padding:.65rem 1.1rem;margin-top:.3rem;font-size:.83rem;
                        color:#78350f;line-height:1.7;">
                <b>Chế độ tự nhập:</b> Tất cả ô hiển thị giá trị 0.0.<br>
                Hãy sửa lại các giá trị theo giao dịch cần kiểm tra.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:.5rem;'></div>", unsafe_allow_html=True)

    pdata = PRESETS[preset_choice]
    def pv(field, default=0.0):
        if pdata is None:
            return default
        return float(pdata.get(field, default))

    with st.form("predict_form"):

        # Time & Amount
        st.markdown('<div class="input-group-title">⏱️ Thông tin chung</div>',
                    unsafe_allow_html=True)
        ct, ca, _, __ = st.columns([2, 2, 1, 1], gap="medium")
        with ct:
            time_val = st.number_input(
                "Time (giây từ giao dịch đầu tiên)",
                value=pv("Time"), step=1.0, format="%.2f",
                help="Khoảng thời gian tính từ giao dịch đầu tiên trong dataset (đơn vị: giây)"
            )
        with ca:
            amount_val = st.number_input(
                "Amount (giá trị giao dịch, USD)",
                value=pv("Amount"), min_value=0.0, step=0.01, format="%.2f",
                help="Giá trị tiền của giao dịch tính bằng USD"
            )

        # V1 – V14
        st.markdown('<div class="input-group-title">🔢 Đặc trưng PCA — V1 đến V14</div>',
                    unsafe_allow_html=True)
        v_vals = {}
        r1 = st.columns(7, gap="small")
        for i, idx in enumerate(range(1, 8)):
            with r1[i]:
                v_vals[f"V{idx}"] = st.number_input(
                    f"V{idx}", value=pv(f"V{idx}"), format="%.4f", key=f"v{idx}",
                    help=f"Thành phần PCA thứ {idx} (đã ẩn danh hoá)"
                )
        r2 = st.columns(7, gap="small")
        for i, idx in enumerate(range(8, 15)):
            with r2[i]:
                v_vals[f"V{idx}"] = st.number_input(
                    f"V{idx}", value=pv(f"V{idx}"), format="%.4f", key=f"v{idx}",
                    help=f"Thành phần PCA thứ {idx} (đã ẩn danh hoá)"
                )

        # V15 – V28
        st.markdown('<div class="input-group-title">🔢 Đặc trưng PCA — V15 đến V28</div>',
                    unsafe_allow_html=True)
        r3 = st.columns(7, gap="small")
        for i, idx in enumerate(range(15, 22)):
            with r3[i]:
                v_vals[f"V{idx}"] = st.number_input(
                    f"V{idx}", value=pv(f"V{idx}"), format="%.4f", key=f"v{idx}",
                    help=f"Thành phần PCA thứ {idx} (đã ẩn danh hoá)"
                )
        r4 = st.columns(7, gap="small")
        for i, idx in enumerate(range(22, 29)):
            with r4[i]:
                v_vals[f"V{idx}"] = st.number_input(
                    f"V{idx}", value=pv(f"V{idx}"), format="%.4f", key=f"v{idx}",
                    help=f"Thành phần PCA thứ {idx} (đã ẩn danh hoá)"
                )

        submitted = st.form_submit_button(
            "🚀  Chạy dự đoán",
            use_container_width=True,
            type="primary",
        )

    # ════════════════════════════════════════
    # PHẦN 2 — SƠ ĐỒ PIPELINE
    # ════════════════════════════════════════
    st.markdown('<div class="section-header">🔄 Pipeline xử lý dữ liệu</div>',
                unsafe_allow_html=True)

    def draw_pipeline(done=False):
        steps = [
            ("📥", "Nhập\ndữ liệu thô",     "30 đặc trưng\nTime · V1–V28\nAmount (USD)"),
            ("⚙️",  "Tiền xử lý\nScaler",    "StandardScaler\nnormalize\nTime & Amount"),
            ("🔀",  "Tạo vector\nđặc trưng", "numpy array\nshape (1, 30)\ncho inference"),
            ("🤖",  "Dự đoán\nmô hình",      "Logistic Reg.\n        &\nXGBoost"),
            ("🎯",  "Kết quả\ncuối cùng",    "Class 0 / 1\n+ Xác suất\n(probability)"),
        ]
        n = len(steps)
        FW = 13.2
        fig, ax = plt.subplots(figsize=(FW, 3.2), facecolor="#f8fafc")
        ax.set_xlim(0, FW); ax.set_ylim(0, 3.2); ax.axis("off")
        fig.patch.set_facecolor("#f8fafc")

        BW, BH = 2.05, 2.05
        GAP = (FW - n * BW) / (n + 1)
        xs  = [GAP + i * (BW + GAP) + BW / 2 for i in range(n)]
        CY  = 1.65

        DONE_BC, DONE_FC, DONE_TC = "#10b981", "#d1fae5", "#065f46"
        IDLE_BC, IDLE_FC, IDLE_TC = "#cbd5e1", "#f8fafc",  "#94a3b8"
        bc_list = [DONE_BC if done else IDLE_BC] * n
        fc_list = [DONE_FC if done else IDLE_FC] * n
        tc_list = [DONE_TC if done else IDLE_TC] * n

        import matplotlib.patches as mpatches
        for i, (icon, title, sub) in enumerate(steps):
            bc, fc, tc = bc_list[i], fc_list[i], tc_list[i]

            fancy = mpatches.FancyBboxPatch(
                (xs[i] - BW/2 + .06, CY - BH/2),
                BW - .12, BH,
                boxstyle="round,pad=0.05",
                linewidth=2.2, edgecolor=bc, facecolor=fc, zorder=3
            )
            ax.add_patch(fancy)

            circ = plt.Circle((xs[i] - BW/2 + .3, CY + BH/2 - .22),
                               0.19, color=bc, zorder=5)
            ax.add_patch(circ)
            ax.text(xs[i] - BW/2 + .3, CY + BH/2 - .22, str(i+1),
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="white", zorder=6)

            if done:
                ax.text(xs[i] + BW/2 - .32, CY + BH/2 - .22, "✓",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color=DONE_BC, zorder=6)

            ax.text(xs[i], CY + .45, icon,
                    ha="center", va="center", fontsize=18, zorder=4)
            ax.text(xs[i], CY - .07, title,
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color=tc, zorder=4, linespacing=1.45)
            ax.text(xs[i], CY - .67, sub,
                    ha="center", va="center", fontsize=6.8,
                    color="#64748b", zorder=4, linespacing=1.35)

            if i < n - 1:
                xs0 = xs[i]  + BW/2 - .06
                xe0 = xs[i+1] - BW/2 + .06
                arrow_c = DONE_BC if done else IDLE_BC
                ax.annotate("",
                    xy=(xe0, CY), xytext=(xs0, CY),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_c,
                                    lw=2.4, mutation_scale=18),
                    zorder=2)

        plt.tight_layout(pad=0.1)
        return fig

    pipe_fig = draw_pipeline(done=submitted)
    st.markdown('<div class="pipe-wrap">', unsafe_allow_html=True)
    st.pyplot(pipe_fig, use_container_width=True)
    plt.close(pipe_fig)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📖 Xem giải thích chi tiết từng bước"):
        details = [
            ("1️⃣", "Nhập dữ liệu thô",
             "Người dùng điền 30 giá trị: <b>Time</b> (giây), <b>V1–V28</b> (đặc trưng PCA đã ẩn danh hoá), "
             "<b>Amount</b> (USD). Đây là dữ liệu thô chưa qua xử lý, giống hệt format dataset gốc."),
            ("2️⃣", "Tiền xử lý (Scaler)",
             "<b>StandardScaler</b> chuẩn hoá toàn bộ 30 cột về phân phối chuẩn "
             "(mean=0, std=1). Bước này <b>bắt buộc phải giống hệt</b> lúc huấn luyện để tránh data leakage."),
            ("3️⃣", "Tạo vector đặc trưng",
             "Ghép tất cả 30 đặc trưng đã xử lý thành <b>numpy array</b> shape <b>(1, 30)</b> "
             "— đúng định dạng mà model sklearn/xgboost yêu cầu khi gọi <code>predict()</code>."),
            ("4️⃣", "Dự đoán mô hình",
             "<b>Logistic Regression</b>: tính xác suất dựa trên hàm sigmoid của tổ hợp tuyến tính. "
             "<b>XGBoost</b>: ensemble nhiều cây quyết định, mỗi cây sửa lỗi của cây trước "
             "(gradient boosting). Cả hai chạy <b>song song, độc lập</b>."),
            ("5️⃣", "Kết quả cuối cùng",
             "Trả về <b>Class 0</b> (Hợp lệ) hoặc <b>Class 1</b> (Gian lận) kèm <b>xác suất</b> "
             "(0–100%). Hiển thị kết quả từng mô hình riêng và <b>so sánh sự đồng thuận</b> giữa hai mô hình."),
        ]
        dcols = st.columns(5, gap="small")
        for col, (num, ttl, desc) in zip(dcols, details):
            with col:
                st.markdown(f"""
                <div class="step-detail">
                    <b>{num} {ttl}</b><br><br>{desc}
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════
    # PHẦN 3 — PIPELINE CHI TIẾT + KẾT QUẢ
    # ════════════════════════════════════════
    if submitted:

        feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        raw_dict = {"Time": time_val, "Amount": amount_val}
        raw_dict.update(v_vals)
        X_raw = np.array([[raw_dict[f] for f in feature_order]], dtype=float)

        if scaler is not None:
            X_scaled  = scaler.transform(X_raw)
            scaler_ok = True
        else:
            X_scaled  = X_raw.copy()
            scaler_ok = False

        X_input = X_scaled

        def run_predict(model, X):
            if model is None:
                return None, None, None
            try:
                pred  = int(model.predict(X)[0])
                proba = model.predict_proba(X)[0]
                try:
                    z = float(model.decision_function(X)[0])
                except Exception:
                    z = None
                return pred, float(proba[1]), z
            except Exception as e:
                st.error(f"Lỗi: {e}")
                return None, None, None

        pred_lr,  prob_lr,  z_lr  = run_predict(model_lr,  X_input)
        pred_xgb, prob_xgb, z_xgb = run_predict(model_xgb, X_input)

        st.markdown('<div class="section-header">🔬 Chi tiết từng bước xử lý</div>',
                    unsafe_allow_html=True)

        with st.expander("📥  BƯỚC 1 — Dữ liệu đầu vào thô (Raw Input)", expanded=True):
            st.markdown("""
            <div style='font-size:.82rem;color:#475569;margin-bottom:.6rem;'>
            <b>Input:</b> Người dùng nhập 30 giá trị &nbsp;→&nbsp;
            <b>Output:</b> numpy array shape <code>(1, 30)</code> chưa qua xử lý
            </div>
            """, unsafe_allow_html=True)
            raw_display = pd.DataFrame(X_raw, columns=feature_order).round(6)
            st.dataframe(raw_display, use_container_width=True, hide_index=True)
            ci1, ci2, ci3 = st.columns(3)
            ci1.metric("Time (giây)", f"{time_val:,.2f}")
            ci2.metric("Amount (USD)", f"${amount_val:,.2f}")
            ci3.metric("Số đặc trưng", "30")

        with st.expander("⚙️  BƯỚC 2 — Sau StandardScaler (Chuẩn hoá)", expanded=True):
            if scaler_ok:
                st.markdown("""
                <div style='font-size:.82rem;color:#475569;margin-bottom:.6rem;'>
                <b>Input:</b> Array thô (1×30) &nbsp;→&nbsp;
                <b>Xử lý:</b> StandardScaler transform toàn bộ 30 cột (mean=0, std=1) &nbsp;→&nbsp;
                <b>Output:</b> Array đã chuẩn hoá (1×30)
                </div>
                """, unsafe_allow_html=True)
                scaled_display = pd.DataFrame(X_scaled, columns=feature_order).round(6)
                st.dataframe(scaled_display, use_container_width=True, hide_index=True)
                cs1, cs2, cs3, cs4 = st.columns(4)
                cs1.metric("Time  → trước", f"{X_raw[0,0]:,.2f}")
                cs2.metric("Time  → sau",   f"{X_scaled[0,0]:.6f}")
                cs3.metric("Amount → trước", f"{X_raw[0,-1]:,.2f}")
                cs4.metric("Amount → sau",   f"{X_scaled[0,-1]:.6f}")
                st.caption("✅ Tất cả 30 đặc trưng đã được đưa về phân phối chuẩn (mean ≈ 0, std ≈ 1)")
            else:
                st.warning("⚠️ Chưa load được scaler.pkl — hãy điền SCALER_PATH trong code.")

        with st.expander("🔀  BƯỚC 3 — Vector đặc trưng sẵn sàng cho model", expanded=False):
            st.markdown(f"""
            <div style='font-size:.82rem;color:#475569;margin-bottom:.8rem;'>
            <b>Input:</b> Array đã scale (1×30) &nbsp;→&nbsp;
            <b>Output:</b> numpy array shape <code>{X_input.shape}</code>, dtype <code>float64</code>
            — đúng format để truyền vào <code>model.predict()</code>
            </div>
            """, unsafe_allow_html=True)
            vec_df = pd.DataFrame({
                "Đặc trưng":        feature_order,
                "Giá trị thô":      X_raw[0].round(6),
                "Giá trị sau scale": X_input[0].round(6),
            })
            st.dataframe(vec_df, use_container_width=True, hide_index=True, height=320)

        with st.expander("📐  BƯỚC 4a — Logistic Regression: từng bước tính toán", expanded=True):
            if pred_lr is not None and model_lr is not None:
                st.markdown("""
                <div style='font-size:.82rem;color:#1e3a5f;font-weight:700;
                            border-left:3px solid #3b82f6;padding-left:.6rem;margin-bottom:.8rem;'>
                Logistic Regression — Pipeline nội bộ
                </div>
                """, unsafe_allow_html=True)

                coef   = model_lr.coef_[0]
                interc = model_lr.intercept_[0]
                z_score = float(np.dot(coef, X_input[0]) + interc)
                prob_sigmoid = float(1 / (1 + np.exp(-z_score)))

                contributions = coef * X_input[0]
                contrib_df = pd.DataFrame({
                    "Đặc trưng":      feature_order,
                    "Hệ số (coef)":   coef.round(6),
                    "Giá trị input":  X_input[0].round(6),
                    "Đóng góp (w×x)": contributions.round(6),
                }).sort_values("Đóng góp (w×x)", key=abs, ascending=False)

                step_cols = st.columns([1, 1, 1], gap="medium")
                with step_cols[0]:
                    st.markdown("""
                    <div style='background:#eff6ff;border:1.5px solid #3b82f6;border-radius:10px;
                                padding:.9rem 1rem;font-size:.8rem;line-height:1.8;'>
                    <b style='color:#1e3a5f;'>① Tổ hợp tuyến tính</b><br>
                    <code>z = w₁x₁ + w₂x₂ + ... + w₃₀x₃₀ + b</code><br><br>
                    <b>Intercept b =</b> {:.6f}<br>
                    <b>Tổng w·x =</b> {:.6f}<br>
                    <b style='font-size:.9rem;'>z = {:.6f}</b>
                    </div>
                    """.format(interc, z_score - interc, z_score), unsafe_allow_html=True)

                with step_cols[1]:
                    st.markdown("""
                    <div style='background:#f0fdf4;border:1.5px solid #10b981;border-radius:10px;
                                padding:.9rem 1rem;font-size:.8rem;line-height:1.8;'>
                    <b style='color:#065f46;'>② Hàm Sigmoid</b><br>
                    <code>P = 1 / (1 + e^(-z))</code><br><br>
                    <b>z =</b> {:.6f}<br>
                    <b>e^(-z) =</b> {:.6f}<br>
                    <b style='font-size:.9rem;'>P(Fraud) = {:.6f}</b>
                    </div>
                    """.format(z_score, float(np.exp(-z_score)), prob_sigmoid), unsafe_allow_html=True)

                with step_cols[2]:
                    is_fr = pred_lr == 1
                    bg  = "#fef2f2" if is_fr else "#f0fdf4"
                    bdr = "#ef4444" if is_fr else "#10b981"
                    lbl = "🚨 GIAN LẬN (1)" if is_fr else "✅ HỢP LỆ (0)"
                    st.markdown(f"""
                    <div style='background:{bg};border:1.5px solid {bdr};border-radius:10px;
                                padding:.9rem 1rem;font-size:.8rem;line-height:1.8;'>
                    <b style='color:#1e3a5f;'>③ Ngưỡng quyết định</b><br>
                    <code>Class = 1 nếu P &gt; 0.5</code><br><br>
                    <b>P(Fraud) =</b> {prob_sigmoid:.6f}<br>
                    <b>Ngưỡng =</b> 0.5<br>
                    <b style='font-size:.9rem;'>{lbl}</b>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Top 10 đặc trưng có đóng góp lớn nhất vào z-score:**")
                st.dataframe(contrib_df.head(10), use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Chưa load được Logistic Regression model.")

        with st.expander("🌲  BƯỚC 4b — XGBoost: từng bước tính toán", expanded=True):
            if pred_xgb is not None and model_xgb is not None:
                st.markdown("""
                <div style='font-size:.82rem;color:#1e3a5f;font-weight:700;
                            border-left:3px solid #f59e0b;padding-left:.6rem;margin-bottom:.8rem;'>
                XGBoost — Pipeline nội bộ (Gradient Boosted Trees)
                </div>
                """, unsafe_allow_html=True)

                n_trees   = model_xgb.n_estimators
                max_depth = model_xgb.max_depth
                lr_xgb    = model_xgb.learning_rate

                try:
                    import xgboost as xgb
                    dmat      = xgb.DMatrix(X_input, feature_names=feature_order)
                    raw_score = float(model_xgb.get_booster().predict(dmat, output_margin=True)[0])
                    prob_xgb_calc = float(1 / (1 + np.exp(-raw_score)))
                except Exception:
                    raw_score     = None
                    prob_xgb_calc = prob_xgb

                imp_dict = model_xgb.get_booster().get_fscore()
                if imp_dict:
                    imp_df = pd.DataFrame({
                        "Đặc trưng": list(imp_dict.keys()),
                        "F-score":   list(imp_dict.values()),
                    }).sort_values("F-score", ascending=False).head(10)
                else:
                    imp_arr = model_xgb.feature_importances_
                    imp_df = pd.DataFrame({
                        "Đặc trưng": feature_order,
                        "F-score":   imp_arr,
                    }).sort_values("F-score", ascending=False).head(10)

                step_cols2 = st.columns([1, 1, 1], gap="medium")
                with step_cols2[0]:
                    st.markdown(f"""
                    <div style='background:#fffbeb;border:1.5px solid #f59e0b;border-radius:10px;
                                padding:.9rem 1rem;font-size:.8rem;line-height:1.8;'>
                    <b style='color:#78350f;'>① Cấu trúc Ensemble</b><br>
                    <code>F(x) = Σ fₖ(x), k=1..T</code><br><br>
                    <b>Số cây (T):</b> {n_trees}<br>
                    <b>Max depth:</b> {max_depth}<br>
                    <b>Learning rate:</b> {lr_xgb}
                    </div>
                    """, unsafe_allow_html=True)

                with step_cols2[1]:
                    raw_str = f"{raw_score:.6f}" if raw_score is not None else "N/A"
                    st.markdown(f"""
                    <div style='background:#f0fdf4;border:1.5px solid #10b981;border-radius:10px;
                                padding:.9rem 1rem;font-size:.8rem;line-height:1.8;'>
                    <b style='color:#065f46;'>② Raw Score → Sigmoid</b><br>
                    <code>P = 1 / (1 + e^(-score))</code><br><br>
                    <b>Raw score (log-odds):</b><br>
                    {raw_str}<br>
                    <b style='font-size:.9rem;'>P(Fraud) = {prob_xgb_calc:.6f}</b>
                    </div>
                    """, unsafe_allow_html=True)

                with step_cols2[2]:
                    is_fr2 = pred_xgb == 1
                    bg2  = "#fef2f2" if is_fr2 else "#f0fdf4"
                    bdr2 = "#ef4444" if is_fr2 else "#10b981"
                    lbl2 = "🚨 GIAN LẬN (1)" if is_fr2 else "✅ HỢP LỆ (0)"
                    st.markdown(f"""
                    <div style='background:{bg2};border:1.5px solid {bdr2};border-radius:10px;
                                padding:.9rem 1rem;font-size:.8rem;line-height:1.8;'>
                    <b style='color:#1e3a5f;'>③ Ngưỡng quyết định</b><br>
                    <code>Class = 1 nếu P &gt; 0.5</code><br><br>
                    <b>P(Fraud) =</b> {prob_xgb_calc:.6f}<br>
                    <b>Ngưỡng =</b> 0.5<br>
                    <b style='font-size:.9rem;'>{lbl2}</b>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Top 10 đặc trưng quan trọng nhất (F-score — số lần được dùng để split):**")
                st.dataframe(imp_df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Chưa load được XGBoost model.")

        st.markdown('<div class="section-header">🎯 Kết quả dự đoán cuối cùng</div>',
                    unsafe_allow_html=True)

        def render_card(col, pred, prob_fraud, model_label, path_var):
            with col:
                st.markdown('<div class="result-outer">', unsafe_allow_html=True)
                st.markdown(f'<div class="model-badge">🤖 {model_label}</div>',
                            unsafe_allow_html=True)

                if pred is None:
                    st.warning(
                        f"⚠️ Chưa load được **{model_label}**.  \n"
                        f"Hãy điền đường dẫn vào `{path_var}` trong code."
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                is_fraud   = pred == 1
                cls        = "fraud" if is_fraud else "legit"
                icon       = "🚨" if is_fraud else "✅"
                label_txt  = "GIAN LẬN" if is_fraud else "HỢP LỆ"
                prob_legit = 1.0 - prob_fraud
                pf_pct     = prob_fraud * 100
                pl_pct     = prob_legit * 100
                confidence = max(pf_pct, pl_pct)

                st.markdown(f"""
                <div class="result-card {cls}">
                    <div class="result-icon">{icon}</div>
                    <div class="result-label {cls}">{label_txt}</div>
                    <div class="result-prob">
                        Giao dịch này được phân loại là <b>{label_txt}</b><br>
                        với độ tin cậy <b>{confidence:.2f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="prob-bar-wrap">
                    <div class="prob-bar-label">
                        <span>✅ Hợp lệ</span><span><b>{pl_pct:.2f}%</b></span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill-legit" style="width:{pl_pct:.2f}%;"></div>
                    </div>
                </div>
                <div class="prob-bar-wrap" style="margin-top:.6rem;">
                    <div class="prob-bar-label">
                        <span>🚨 Gian lận</span><span><b>{pf_pct:.2f}%</b></span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill-fraud" style="width:{pf_pct:.2f}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("Class dự đoán", "1 — Gian lận" if is_fraud else "0 — Hợp lệ")
                m2.metric("P(Fraud)", f"{pf_pct:.4f}%")
                st.markdown("</div>", unsafe_allow_html=True)

        col_lr, col_xgb = st.columns(2, gap="large")
        render_card(col_lr,  pred_lr,  prob_lr,  "Logistic Regression", "MODEL_LR_PATH")
        render_card(col_xgb, pred_xgb, prob_xgb, "XGBoost",             "MODEL_XGB_PATH")

        if pred_lr is not None and pred_xgb is not None:
            st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚖️ So sánh kết quả hai mô hình</div>',
                        unsafe_allow_html=True)

            agree = pred_lr == pred_xgb
            if agree:
                verdict = "✅ Hợp lệ" if pred_lr == 0 else "🚨 Gian lận"
                st.success(
                    f"**Cả hai mô hình đồng thuận:** Giao dịch này là **{verdict}**  \n"
                    f"LR — P(Fraud): {prob_lr*100:.4f}%  ·  "
                    f"XGBoost — P(Fraud): {prob_xgb*100:.4f}%"
                )
            else:
                st.warning(
                    "⚡ **Hai mô hình không đồng thuận** — cần xem xét thêm.  \n"
                    f"Logistic Regression: **{'Gian lận' if pred_lr==1 else 'Hợp lệ'}**  ·  "
                    f"XGBoost: **{'Gian lận' if pred_xgb==1 else 'Hợp lệ'}**"
                )

            cmp = pd.DataFrame({
                "Mô hình":     ["Logistic Regression", "XGBoost"],
                "Dự đoán":     [
                    "🚨 Gian lận" if pred_lr  == 1 else "✅ Hợp lệ",
                    "🚨 Gian lận" if pred_xgb == 1 else "✅ Hợp lệ",
                ],
                "P(Fraud) %":  [f"{prob_lr*100:.4f}",   f"{prob_xgb*100:.4f}"],
                "P(Hợp lệ) %": [f"{(1-prob_lr)*100:.4f}", f"{(1-prob_xgb)*100:.4f}"],
                "Đồng thuận":  ["✅ Có" if agree else "❌ Không"] * 2,
            })
            st.dataframe(cmp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — ĐÁNH GIÁ & HIỆU NĂNG
# ══════════════════════════════════════════════════════════════════
elif page == "📈  Huấn luyện & Đánh giá":

    # ── Helper hiển thị ảnh ──────────────────────────────────────
    def show_img(path):
        if os.path.exists(path):
            st.image(path, use_column_width=True)
        else:
            st.warning(f"⚠️ Không tìm thấy ảnh tại: `{path}`")

    # ── Hero ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">📈 Trang 3 · Đánh giá & Hiệu năng</div>
        <div class="hero-title">
            Đánh giá toàn diện <span>hai mô hình phân loại</span>
        </div>
        <div class="hero-sub">
            Logistic Regression vs XGBoost — Chứng minh mô hình hoạt động tốt và đáng tin cậy
        </div>
        <hr class="hero-divider">
        <div class="hero-meta">
            <span>📊 <b>12 biểu đồ</b> đánh giá chi tiết</span>
            <span>🎯 <b>Accuracy · F1 · AUC · Precision · Recall</b></span>
            <span>🔬 <b>Phân tích sai số</b> chuyên sâu</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # PHẦN A — CÁC CHỈ SỐ ĐO LƯỜNG
    # ════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📊 A. Các chỉ số đo lường hiệu năng</div>',
                unsafe_allow_html=True)

    # ── Confusion Matrix ─────────────────────────────────────────
    st.markdown('<span class="chart-label">🧮 05 — Confusion Matrix</span>',
                unsafe_allow_html=True)
    show_img(IMG_05)
    ca, cb = st.columns(2, gap="medium")
    with ca:
        st.markdown("""
        <div class="cmt-warn">
        <b>📐 Logistic Regression:</b><br>
        • TP=90 · FP=1,450 · FN=8 · TN=55,414<br>
        • Recall = 0.9184 → phát hiện được 91.8% gian lận ✅<br>
        • Precision = 0.0584 → 100 cảnh báo chỉ ~6 vụ thật 🟡<br><br>
        <b>🔺 Vấn đề:</b> 1,450 False Positive — báo nhầm quá nhiều giao dịch
        hợp lệ, gây phiền toái cho khách hàng và tốn chi phí xử lý.
        </div>
        """, unsafe_allow_html=True)
    with cb:
        st.markdown("""
        <div class="cmt-good">
        <b>📐 XGBoost:</b><br>
        • TP=86 · FP=669 · FN=12 · TN=56,195<br>
        • Recall = 0.8776 → phát hiện được 87.8% gian lận 🟡<br>
        • Precision = 0.1139 → 100 cảnh báo ~11 vụ thật ✅<br><br>
        <b>✅ Cải thiện rõ rệt:</b> FP giảm từ 1,450 xuống 669 (giảm 54%).
        Precision gần gấp đôi LR — đáng đánh đổi một chút Recall
        trong thực tế triển khai.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    # ── ROC & PR Curve ───────────────────────────────────────────
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<span class="chart-label">📈 06 — ROC Curve</span>',
                    unsafe_allow_html=True)
        show_img(IMG_06)
        st.markdown("""
        <div class="cmt-good">
        <b>✅ Kết quả xuất sắc:</b><br>
        • LR: AUC = <b>0.9714</b><br>
        • XGBoost: AUC = <b>0.9836</b><br><br>
        Cả hai đều vượt ngưỡng <b>0.97</b> — rất tốt trong thực tế.
        XGBoost nhỉnh hơn LR <b>+1.22% AUC</b>, đặc biệt ở vùng
        FPR thấp (< 0.1) — vùng quan trọng nhất khi triển khai.
        AUC > 0.98 của XGBoost đạt mức <b>"Excellent"</b>.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<span class="chart-label">🎯 07 — Precision-Recall Curve</span>',
                    unsafe_allow_html=True)
        show_img(IMG_07)
        st.markdown("""
        <div class="cmt-good">
        <b>✅ Kết quả:</b><br>
        • LR: AP = <b>0.7133</b><br>
        • XGBoost: AP = <b>0.8322</b><br><br>
        XGBoost vượt trội LR đến <b>+11.9%</b> Average Precision.
        Đường PR của XGBoost giữ Precision ≈ 1.0 cho đến Recall ≈ 0.8
        — tức là phát hiện 80% gian lận với độ chính xác gần tuyệt đối.
        LR sụt Precision sớm hơn từ Recall ~0.4.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    # ── Metrics comparison ───────────────────────────────────────
    st.markdown('<span class="chart-label">⚖️ 08 — So sánh toàn diện các chỉ số</span>',
                unsafe_allow_html=True)
    show_img(IMG_08)
    st.markdown("""
    <div class="cmt-good">
    <b>📊 Tổng hợp so sánh:</b><br><br>
    <table style="width:100%;font-size:.82rem;border-collapse:collapse;">
    <tr style="background:#1e3a5f;color:white;">
        <th style="padding:.4rem .7rem;text-align:left;">Chỉ số</th>
        <th style="padding:.4rem .7rem;text-align:center;">Logistic Regression</th>
        <th style="padding:.4rem .7rem;text-align:center;">XGBoost</th>
        <th style="padding:.4rem .7rem;text-align:center;">Nhận xét</th>
    </tr>
    <tr style="background:#f8fafc;">
        <td style="padding:.35rem .7rem;">Accuracy</td>
        <td style="text-align:center;">0.974</td>
        <td style="text-align:center;font-weight:700;">0.988</td>
        <td style="text-align:center;color:#10b981;">XGB +1.4%</td>
    </tr>
    <tr>
        <td style="padding:.35rem .7rem;">Balanced Accuracy</td>
        <td style="text-align:center;font-weight:700;">0.946</td>
        <td style="text-align:center;">0.933</td>
        <td style="text-align:center;color:#f59e0b;">LR +1.3%</td>
    </tr>
    <tr style="background:#f8fafc;">
        <td style="padding:.35rem .7rem;">Precision (Fraud)</td>
        <td style="text-align:center;">0.058</td>
        <td style="text-align:center;font-weight:700;">0.114</td>
        <td style="text-align:center;color:#10b981;">XGB +96%</td>
    </tr>
    <tr>
        <td style="padding:.35rem .7rem;">Recall (Fraud)</td>
        <td style="text-align:center;font-weight:700;">0.918</td>
        <td style="text-align:center;">0.878</td>
        <td style="text-align:center;color:#f59e0b;">LR +4.6%</td>
    </tr>
    <tr style="background:#f8fafc;">
        <td style="padding:.35rem .7rem;">F1-score (Fraud)</td>
        <td style="text-align:center;">0.110</td>
        <td style="text-align:center;font-weight:700;">0.202</td>
        <td style="text-align:center;color:#10b981;">XGB +84%</td>
    </tr>
    <tr>
        <td style="padding:.35rem .7rem;">ROC-AUC</td>
        <td style="text-align:center;">0.971</td>
        <td style="text-align:center;font-weight:700;">0.984</td>
        <td style="text-align:center;color:#10b981;">XGB +1.3%</td>
    </tr>
    </table><br>
    <b>🏆 Kết luận:</b> XGBoost vượt trội trên hầu hết chỉ số, đặc biệt F1 và Precision.
    LR có Recall và Balanced Accuracy cao hơn nhẹ — phù hợp khi ưu tiên bắt tối đa
    gian lận (chấp nhận báo nhầm nhiều). XGBoost phù hợp triển khai thực tế hơn.
    </div>
    """, unsafe_allow_html=True)

    # ── Classification Report ────────────────────────────────────
    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="chart-label">📋 16 — Classification Report (Heatmap)</span>',
                unsafe_allow_html=True)
    show_img(IMG_16)
    st.markdown("""
    <div class="cmt-improve">
    <b>✅ Đọc kết quả:</b> Lớp <b>Hợp lệ</b> có precision/recall/F1 > 0.99 ở cả 2 mô hình — rất tốt.
    Lớp <b>Gian lận</b> là thách thức chính: F1 chỉ đạt 0.110 (LR) và 0.202 (XGB).<br><br>
    <b>🔺 Tại sao F1 Fraud thấp?</b> Do Precision thấp (nhiều FP), không phải do Recall yếu.
    Đây là hệ quả trực tiếp của dữ liệu cực mất cân bằng (0.17% fraud).<br><br>
    <b>💡 Cải thiện F1 Fraud:</b> Tăng ngưỡng quyết định (xem biểu đồ 14),
    hoặc thử Cost-sensitive learning với class_weight tuỳ chỉnh.
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # PHẦN B — BIỂU ĐỒ KỸ THUẬT
    # ════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🔧 B. Biểu đồ kỹ thuật</div>',
                unsafe_allow_html=True)

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        st.markdown('<span class="chart-label">🌲 09 — XGBoost Feature Importance (3 loại)</span>',
                    unsafe_allow_html=True)
        show_img(IMG_09)
        st.markdown("""
        <div class="cmt-good">
        <b>✅ Phân tích 3 loại importance:</b><br>
        • <b>Weight:</b> V4 xuất hiện >1,300 lần split → đặc trưng dùng phổ biến nhất.<br>
        • <b>Gain:</b> V14 có gain ~3,200 → mỗi lần split bằng V14 cải thiện
        loss nhiều nhất — <b>quan trọng nhất về chất lượng</b>.<br>
        • <b>Cover:</b> V3 ảnh hưởng đến ~360K sample mỗi split.<br><br>
        <b>🏆 Đặc trưng then chốt:</b> V4, V14, V1, V10 — nhất quán
        trong cả 3 loại, khớp với correlation heatmap ở trang 1.
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown('<span class="chart-label">📐 10 — Logistic Regression Coefficients</span>',
                    unsafe_allow_html=True)
        show_img(IMG_10)
        st.markdown("""
        <div class="cmt-improve">
        <b>✅ Phân tích hệ số LR:</b><br>
        • <b>Tăng xác suất gian lận (đỏ):</b> V4 (+1.60), V1 (+1.05),
        Amount (+0.87), V5 (+0.74).<br>
        • <b>Giảm xác suất gian lận (xanh):</b> V10 (-1.45), V14 (-1.30),
        V17 (-1.20).<br><br>
        <b>💡 So sánh:</b> V14 và V10 quan trọng ở cả LR lẫn XGBoost →
        xác nhận đây là đặc trưng phân biệt thực sự.<br><br>
        <b>🔺 Hạn chế LR:</b> Chỉ nắm quan hệ tuyến tính, bỏ qua
        tương tác phức tạp mà XGBoost xử lý được.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    c5, c6 = st.columns(2, gap="medium")
    with c5:
        st.markdown('<span class="chart-label">📉 11 — Learning Curve</span>',
                    unsafe_allow_html=True)
        show_img(IMG_11)
        st.markdown("""
        <div class="cmt-good">
        <b>✅ Logistic Regression:</b> Train F1 và Val F1 hội tụ tốt khi
        training size > 150K, không overfitting rõ ràng. Mô hình vẫn
        cải thiện khi có thêm dữ liệu (underfitting nhẹ).<br><br>
        <b>✅ XGBoost:</b> Val F1 đạt ~0.99 từ 50K samples và ổn định —
        học <b>rất nhanh và hiệu quả</b>. Train F1 ≈ 1.0 cho thấy fit
        tốt trên tập train.<br><br>
        <b>💡 Kết luận:</b> XGBoost phù hợp hơn khi dataset nhỏ.
        LR cần nhiều dữ liệu hơn để đạt hiệu năng tốt.
        </div>
        """, unsafe_allow_html=True)

    with c6:
        st.markdown('<span class="chart-label">📊 12 — XGBoost Training History</span>',
                    unsafe_allow_html=True)
        show_img(IMG_12)
        st.markdown("""
        <div class="cmt-warn">
        <b>✅ Quá trình huấn luyện ổn định:</b> Val AUC tăng đều từ 0.70
        lên 0.984, không có dấu hiệu bất ổn hay dao động.<br><br>
        <b>🔺 Khoảng cách Train-Val:</b> Train AUC ≈ 1.0, Val AUC ≈ 0.984
        → có <b>overfitting nhẹ</b> (~1.6%). Có thể giảm bằng cách
        tăng regularization (giảm max_depth, tăng min_child_weight).<br><br>
        <b>💡 Best epoch = 234</b> — tăng n_estimators lên 300 vẫn
        cải thiện nhưng marginal gain từ epoch 200+ rất nhỏ.
        Có thể dùng n_estimators=234 để tiết kiệm thời gian inference.
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # PHẦN C — PHÂN TÍCH CHUYÊN SÂU
    # ════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🔬 C. Phân tích chuyên sâu</div>',
                unsafe_allow_html=True)

    c7, c8 = st.columns(2, gap="medium")
    with c7:
        st.markdown('<span class="chart-label">📦 13 — Phân phối xác suất P(Fraud)</span>',
                    unsafe_allow_html=True)
        show_img(IMG_13)
        st.markdown("""
        <div class="cmt-good">
        <b>✅ Phân tách rõ ràng:</b> Cả 2 mô hình đều tách tốt hai lớp:<br>
        • Giao dịch hợp lệ → P(Fraud) tập trung gần <b>0</b><br>
        • Giao dịch gian lận → P(Fraud) tập trung gần <b>1</b><br><br>
        <b>✅ XGBoost phân tách sắc nét hơn:</b> Peak của lớp hợp lệ
        tại P≈0 cao và hẹp hơn LR, ít overlap hơn ở vùng trung gian.<br><br>
        <b>💡</b> Điều này giải thích tại sao XGBoost có AUC cao hơn
        — xác suất được "phân tách" tốt hơn về 2 cực.
        </div>
        """, unsafe_allow_html=True)

    with c8:
        st.markdown('<span class="chart-label">⚙️ 14 — Threshold Analysis</span>',
                    unsafe_allow_html=True)
        show_img(IMG_14)
        st.markdown("""
        <div class="cmt-warn">
        <b>🔺 Vấn đề với ngưỡng 0.5 mặc định:</b> Ở threshold=0.5,
        F1 và Precision đều rất thấp (< 0.2).<br><br>
        <b>💡 Tối ưu threshold theo mục tiêu:</b><br>
        • Ưu tiên bắt nhiều gian lận (Recall cao) → threshold thấp (~0.3)<br>
        • Ưu tiên ít báo nhầm (Precision cao) → threshold cao (~0.7–0.9)<br><br>
        <b>⚙️ Đề xuất thực tế:</b> Thử threshold = <b>0.5–0.7</b> để
        cân bằng Precision và Recall tốt hơn ngưỡng 0.5 mặc định.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    st.markdown('<span class="chart-label">🔎 15 — Phân tích sai số (Error Analysis)</span>',
                unsafe_allow_html=True)
    show_img(IMG_15)
    st.markdown("""
    <div class="cmt-improve">
    <b>🔺 False Negative (bỏ sót gian lận) — nguy hiểm nhất:</b><br>
    • LR: chỉ 8 vụ FN — P(Fraud) của những vụ này rất thấp (< 0.1)
    → mô hình không hề "nghi ngờ" → khó cải thiện hơn.<br>
    • XGB: 12 vụ FN — tương tự, cần feature engineering thêm để
    phân biệt những case đặc biệt này.<br><br>
    <b>🔺 False Positive (báo nhầm hợp lệ) — tốn kém trong thực tế:</b><br>
    • LR: <b>1,450 FP</b> với P(Fraud) rải đều 0.5–1.0 → mô hình
    không chắc nhưng vẫn báo.<br>
    • XGB: <b>669 FP</b> (giảm 54%) → tốt hơn rõ rệt.<br><br>
    <b>💡 Hướng cải thiện:</b><br>
    ① Tăng threshold từ 0.5 lên 0.6–0.7 để giảm FP đáng kể.<br>
    ② Feature engineering: thêm velocity features (số giao dịch trong 1 giờ qua).<br>
    ③ Thử Isolation Forest hoặc Autoencoder để phát hiện bất thường độc lập.<br>
    ④ Ensemble LR + XGBoost: dùng Voting để kết hợp Recall cao của LR
    và Precision cao của XGB.
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # PHẦN D — KẾT LUẬN & HƯỚNG CẢI THIỆN
    # ════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🏆 D. Kết luận & Hướng cải thiện</div>',
                unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2, gap="large")
    with col_d1:
        st.markdown("""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
                    padding:1.5rem 1.6rem;box-shadow:0 2px 12px rgba(0,0,0,.05);">
        <div style="font-size:.92rem;font-weight:800;color:#1e3a5f;margin-bottom:.9rem;
                    border-bottom:2px solid #3b82f6;padding-bottom:.5rem;">
            🏆 Mô hình được khuyến nghị: XGBoost
        </div>
        <div style="font-size:.83rem;color:#334155;line-height:1.9;">
            ✅ ROC-AUC = <b>0.9836</b> — đạt mức Excellent<br>
            ✅ F1-score Fraud = <b>0.202</b> — gấp đôi Logistic Regression<br>
            ✅ Precision = <b>0.1139</b> — ít báo nhầm hơn LR 96%<br>
            ✅ False Positive chỉ 669 so với 1,450 của LR<br>
            ✅ Learning curve hội tụ nhanh ngay từ ít dữ liệu<br>
            ✅ Xử lý tốt quan hệ phi tuyến giữa các đặc trưng
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
                    padding:1.5rem 1.6rem;box-shadow:0 2px 12px rgba(0,0,0,.05);">
        <div style="font-size:.92rem;font-weight:800;color:#1e3a5f;margin-bottom:.9rem;
                    border-bottom:2px solid #f59e0b;padding-bottom:.5rem;">
            💡 Hướng cải thiện trong tương lai
        </div>
        <div style="font-size:.83rem;color:#334155;line-height:1.9;">
            🔧 <b>Threshold tuning:</b> Điều chỉnh ngưỡng theo yêu cầu nghiệp vụ<br>
            🔧 <b>Feature engineering:</b> Thêm velocity, time-based features<br>
            🔧 <b>Ensemble:</b> Kết hợp LR + XGBoost bằng Stacking/Voting<br>
            🔧 <b>Hyperparameter tuning:</b> GridSearch/Optuna cho XGBoost<br>
            🔧 <b>Anomaly detection:</b> Thử Isolation Forest, Autoencoder<br>
            🔧 <b>Real-time pipeline:</b> Triển khai với stream processing
        </div>
        </div>
        """, unsafe_allow_html=True)
