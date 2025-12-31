import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

# Load dataset
dataset1 = pd.read_csv(
    r"C:\Users\santo\OneDrive\Desktop\Data science\Dec2025\30 Dec 2025\2.LOGISTIC REGRESSION CODE\final1.csv"
)

d2 = dataset1.copy()

# Features & target
X = dataset1.iloc[:, [2, 3]].copy()
y = dataset1.iloc[:, -1]

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        X.loc[:, col] = LabelEncoder().fit_transform(X[col])

# Encode target (IMPORTANT for multiclass AUC)
y = LabelEncoder().fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model (NO deprecated warning)
classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
classifier.fit(X_train, y_train)

# Save predictions on full data
X_scaled_full = sc.fit_transform(X)
d2['y_pred1'] = classifier.predict(X_scaled_full)
d2.to_csv('final1.csv', index=False)

print("✅ Model trained, prediction done & file saved successfully")

# ROC-AUC (multiclass)
y_pred_prob = classifier.predict_proba(X_test)

auc_score = roc_auc_score(
    y_test,
    y_pred_prob,
    multi_class='ovr',
    average='macro'
)

print("ROC AUC Score:", auc_score)
