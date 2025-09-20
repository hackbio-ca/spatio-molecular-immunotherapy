import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -------------------
# Load data
# -------------------
df = pd.read_csv("../../data/Hatem/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")  # change path if needed

# Separate features and target
X = df.drop(columns=["Diabetes_binary"])
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
# Balance classes (if needed)
# -------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# -------------------
# Train model
# -------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_res, y_train_res)

# -------------------
# Evaluate
# -------------------
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
