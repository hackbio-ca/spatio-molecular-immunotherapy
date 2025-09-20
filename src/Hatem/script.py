import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

t_I = time.time()

# Load the csv file as a dataframe
df = pd.read_csv('../../data/Hatem/leukemia_gene_expression.csv')

# Separate features and target (diagnosis is the target)
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

print(y.value_counts())

# Encode diagnosis
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------------------
# Model 1: Basic RandomForest
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=1, stratify=y_encoded
)

clf = RandomForestClassifier(n_estimators=200, random_state=1)
clf.fit(X_train, y_train)

print("Model 1 Evaluation - Basic RandomForest:\n")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred))


# -----------------------------------
# Model 2: Feature Selection + Weights
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=1, stratify=y_encoded
)

# Fit selector ONLY on training data
selector = SelectKBest(score_func=f_classif, k=100)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)

clf2 = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",  # penalize ignoring Healthy
    random_state=1
)
clf2.fit(X_train_new, y_train)

print("\nModel 2 Evaluation - Feature Selection + Balanced RF:\n")
y_pred2 = clf2.predict(X_test_new)
print(classification_report(y_test, y_pred2, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred2))
# Still predicting that 0 are healthy. Might be because I am using a random forest and a random forest here is not a
# good idea?


t_E = time.time()
print(f"\nTotal runtime: {round(t_E - t_I, 3)} s")
