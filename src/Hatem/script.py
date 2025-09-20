import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix



t_I = time.time()

# Load the csv file as a dataframe
df = pd.read_csv('../../data/Hatem/leukemia_gene_expression.csv')

# print(df.head() # Check the dataframe

# Separate features and target (diagnosis is the target)
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# print(X.shape, y.value_counts()) # Checking the data
# print(df.info())

# Encode diagnosis
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0, stratify=y_encoded)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

t_E = time.time()

t_delta = t_E - t_I
print(str(round(t_delta, 3)), 's')


# Evaluate model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred))