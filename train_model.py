"""Latih ulang model Random Forest terbaik (n_estimators=500, max_depth=10) dan simpan ke rf_churn_model.joblib.
Jalankan: python train_model.py  (memakai versi scikit-learn yang terpasang di komputer Anda)"""
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train(csv="telco.csv"):
    df = pd.read_csv(csv)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns="customerID")
    y = df["Churn"].map({"No": 0, "Yes": 1})
    feats = df.drop(columns="Churn")
    cat = [c for c in feats.columns if feats[c].dtype == "object" or str(feats[c].dtype) in ("str", "string")]
    X = pd.get_dummies(feats, columns=cat, drop_first=True).astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    return {"model": model, "columns": X.columns.tolist(), "test_acc": acc}

if __name__ == "__main__":
    art = train()
    joblib.dump(art, "rf_churn_model.joblib", compress=5)
    print("Model disimpan. Akurasi testing: %.4f" % art["test_acc"])
