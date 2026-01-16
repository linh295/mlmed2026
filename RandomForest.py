import zipfile
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
with zipfile.ZipFile("mitbih_train.csv.zip", 'r') as z:
    z.extractall(".")

with zipfile.ZipFile("mitbih_test.csv.zip", 'r') as z:
    z.extractall(".")

train_df = pd.read_csv("mitbih_train.csv", header=None)
test_df  = pd.read_csv("mitbih_test.csv", header=None)
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"   # rất quan trọng cho dữ liệu imbalance
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
