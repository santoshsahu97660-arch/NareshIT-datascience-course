import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\santo\OneDrive\Desktop\Data science\Dec2025\30 Dec 2025\2.LOGISTIC REGRESSION CODE\final1.csv")

print("Dataset Loaded Successfully")
print(dataset.head())
print(dataset.info())

if 'User ID' in dataset.columns:
    dataset = dataset.drop(columns=['User ID'])
    print("\nUser ID column dropped")


X = dataset.iloc[:, :-1]   # Independent variables
y = dataset.iloc[:, -1]    # Dependent variable

print("\nFeatures (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())


from sklearn.preprocessing import LabelEncoder

cat_cols = X.select_dtypes(include='object').columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"{col} encoded as: {list(le.classes_)}")


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

print("\nTrain-Test Split Done")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

print("\nFeature Scaling Done")
print("Train Mean:", X_train.mean(axis=0))
print("Train Std :", X_train.std(axis=0))


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

print("\nModel Trained Successfully")
print("Coefficients:", model.coef_)
print("Intercept  :", model.intercept_)


y_pred = model.predict(X_test)

print("\nActual Values :", y_test.values)
print("Predicted     :", y_pred)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


new_data = [[1, 45, 60000]]
new_data_scaled = sc.transform(new_data)

print("\nNew Input Prediction:")
print("Class :", model.predict(new_data_scaled))
print("Probability :", model.predict_proba(new_data_scaled))

print("\n--- FINAL VALIDATION COMPLETE ---")
