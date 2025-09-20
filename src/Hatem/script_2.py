import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import numpy as np

# -------------------
# Load data
# -------------------
df = pd.read_csv("../../data/Hatem/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")  # change path if needed

# Drop features that are less important or redundant
X = df.drop(columns=["Diabetes_binary", "CholCheck", "AnyHealthcare"])
y = df["Diabetes_binary"]

# -------------------
# Train/test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------
# Scale numeric features
# -------------------
numeric_cols = ["BMI", "Age", "MentHlth", "PhysHlth"]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# -------------------
# Balance classes (SMOTE)
# -------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# -------------------
# Train XGBoost model
# -------------------
xgb_clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric='logloss'
)

xgb_clf.fit(X_train_res, y_train_res)

# -------------------
# Evaluate model
# -------------------
y_pred = xgb_clf.predict(X_test)
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))