import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. LOAD DATA ───────────────────────────────────────────
df = pd.read_csv("creditcard.csv")  # Download from Kaggle
print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")

# ── 2. PREPROCESS ──────────────────────────────────────────
# 'Amount' is the only feature not yet scaled (V1-V28 are already PCA-transformed)
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df.drop(columns=["Amount", "Time"], inplace=True)

X = df.drop(columns=["Class"])
y = df["Class"]

# ── 3. SPLIT ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. HANDLE IMBALANCE WITH SMOTE ─────────────────────────
# Only apply SMOTE on training data to avoid data leakage
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — Fraud: {y_train_res.sum()}, Normal: {(y_train_res==0).sum()}")

# ── 5. TRAIN MODEL ─────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",  # extra safety on top of SMOTE
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_res, y_train_res)

# ── 6. EVALUATE ────────────────────────────────────────────
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# ── 7. CONFUSION MATRIX PLOT ───────────────────────────────
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# ── 8. FEATURE IMPORTANCE ──────────────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind="barh", title="Top 10 Feature Importances")
plt.tight_layout()
plt.show()