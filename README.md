# TA-06 Random Forest: Prediksi Customer Churn
Dataset: Telco Customer Churn (Kaggle: blastchar/telco-customer-churn, 7.043 baris, 21 kolom).

## Isi
- `TA06_RandomForest_Churn.ipynb`: preprocessing, Skenario A (n_estimators), Skenario B (max_depth), feature importance, evaluasi akhir
- `app.py`: deployment web (Streamlit), `rf_churn_model.joblib`: model terlatih
- `fig_*.png`: grafik hasil, `telco.csv`: data

## Menjalankan
```
pip install -r requirements.txt
streamlit run app.py
```
Deploy gratis: push ke GitHub lalu hubungkan ke Streamlit Community Cloud (main file: `app.py`).

## Hasil ringkas
Konfigurasi terbaik n_estimators=500, max_depth=10: akurasi testing 80,3%. Fitur terpenting: tenure, TotalCharges, MonthlyCharges.
