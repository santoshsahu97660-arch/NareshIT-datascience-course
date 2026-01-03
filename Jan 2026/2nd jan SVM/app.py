import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, mean_squared_error, r2_score
)

st.set_page_config(page_title="Smart SVM App", layout="wide")
st.title("🤖 Smart SVM App – Auto Classification / Regression")

file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    target = st.selectbox("🎯 Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)

    # ===============================
    # AUTO DETECT PROBLEM TYPE
    # ===============================
    problem_type = "classification"
    if y.dtype != 'object' and y.nunique() > 10:
        problem_type = "regression"

    st.success(f"🧠 Detected Problem Type: **{problem_type.upper()}**")

    # ===============================
    # Train-Test + Scaling
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kernel = st.selectbox("⚙️ Select SVM Kernel", ["linear", "rbf", "poly", "sigmoid"])

    # ===============================
    # CLASSIFICATION (SVC)
    # ===============================
    if problem_type == "classification":
        model = SVC(kernel=kernel, probability=True)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"✅ Accuracy: {acc:.4f}")

        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.subheader("📝 Classification Report")
        st.text(classification_report(y_test, y_pred))

        if len(np.unique(y)) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            st.subheader("📉 ROC Curve")
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax2.plot([0, 1], [0, 1], linestyle="--")
            ax2.legend()
            st.pyplot(fig2)

    # ===============================
    # REGRESSION (SVR)
    # ===============================
    else:
        model = SVR(kernel=kernel)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success(f"📉 MSE: {mse:.4f}")
        st.success(f"📈 R² Score: {r2:.4f}")

        st.subheader("📊 Actual vs Predicted")
        fig3, ax3 = plt.subplots()
        ax3.scatter(y_test, y_pred)
        ax3.set_xlabel("Actual")
        ax3.set_ylabel("Predicted")
        st.pyplot(fig3)

    st.info("🎉 Auto SVM Analysis Completed Successfully!")
