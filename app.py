import joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="Prediksi Churn Pelanggan Telco", page_icon="📉", layout="centered")
@st.cache_resource
def load_artifact():
    try:
        return joblib.load("rf_churn_model.joblib")
    except Exception:  # versi scikit-learn/numpy berbeda -> latih ulang otomatis
        from train_model import train
        return train()

art = load_artifact()
model, cols = art["model"], art["columns"]

st.title("📉 Prediksi Churn Pelanggan Telco")
st.caption("Random Forest (n_estimators=500, max_depth=10) | TA-06 Pembelajaran Mesin")

c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Lama berlangganan (bulan)", 0, 72, 12)
    monthly = st.number_input("Tagihan bulanan (MonthlyCharges)", 18.0, 120.0, 70.0)
    contract = st.selectbox("Kontrak", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Metode Pembayaran", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
with c2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Punya Partner", ["No", "Yes"])
    dependents = st.selectbox("Punya Tanggungan", ["No", "Yes"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    techsup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
threshold = st.slider("Threshold churn", 0.2, 0.7, 0.5, 0.05,
                      help="Turunkan threshold untuk menangkap lebih banyak calon churn (recall naik, precision turun).")

def build_row():
    r = dict.fromkeys(cols, 0.0)
    r.update({"SeniorCitizen": 1.0 if senior == "Yes" else 0.0, "tenure": float(tenure),
              "MonthlyCharges": monthly, "TotalCharges": tenure * monthly})
    flags = {f"gender_{gender}": 1, f"Partner_{partner}": 1, f"Dependents_{dependents}": 1,
             f"Contract_{contract}": 1, f"InternetService_{internet}": 1, f"PaymentMethod_{payment}": 1,
             f"PaperlessBilling_{paperless}": 1, f"OnlineSecurity_{security}": 1, f"TechSupport_{techsup}": 1}
    for k, v in flags.items():
        if k in r: r[k] = float(v)
    return pd.DataFrame([r])[cols]

if st.button("Prediksi", type="primary"):
    p = float(model.predict_proba(build_row())[0, 1])
    st.metric("Probabilitas churn", f"{p:.1%}")
    st.progress(min(p, 1.0))
    if p >= threshold:
        st.error("⚠️ Berisiko CHURN. Rekomendasi: tawarkan migrasi ke kontrak 1-2 tahun, autopay, atau diskon retensi.")
    else:
        st.success("✅ Kemungkinan bertahan.")
    st.caption("Catatan: TotalCharges diestimasi dari tenure × tagihan bulanan; layanan lain diasumsikan 'No'.")
