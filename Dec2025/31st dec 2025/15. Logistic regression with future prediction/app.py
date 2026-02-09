import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
    roc_curve,
    roc_auc_score
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ML Auto Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ML Auto Analyzer – Final Version")
st.caption("No Errors | Auto Encoding | Confusion Matrix | ROC")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # AUTO EDA
    # --------------------------------------------------
    st.subheader("📊 Auto EDA")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape:", df.shape)
        st.write("Missing Values")
        st.write(df.isnull().sum())

    with col2:
        st.write("Statistical Summary")
        st.write(df.describe(include="all"))

    # --------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------
    st.subheader("⚙️ Model Configuration")

    features = st.multiselect(
        "Select Independent Variables (X)",
        df.columns.tolist()[:-1]
    )

    target = st.selectbox(
        "Select Target Variable (y)",
        df.columns.tolist()
    )

    model_type = st.selectbox(
        "Select Model",
        ["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest"]
    )

    if features and target:

        X = df[features]
        y = df[target]

        # --------------------------------------------------
        # TARGET ENCODING (CRITICAL FIX)
        # --------------------------------------------------
        le = None
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        # --------------------------------------------------
        # AUTO NUMERIC / CATEGORICAL FOR X
        # --------------------------------------------------
        num_features = X.select_dtypes(include=["int64", "float64"]).columns
        cat_features = X.select_dtypes(include=["object"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]), num_features),

                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_features)
            ]
        )

        # --------------------------------------------------
        # MODEL SELECTION
        # --------------------------------------------------
        if model_type == "Linear Regression":
            model = LinearRegression()
            task = "regression"
        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
            task = "classification"
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier()
            task = "classification"
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            task = "classification"

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # --------------------------------------------------
        # TRAIN TEST SPLIT
        # --------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)

        # --------------------------------------------------
        # PREDICTIONS
        # --------------------------------------------------
        y_pred = pipeline.predict(X_test)

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------
        if task == "regression":
            st.subheader("📈 Regression Results")
            st.metric("R² Score", round(r2_score(y_test, y_pred), 3))
            st.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))

        else:
            st.subheader("📈 Classification Results")

            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", round(acc, 3))

            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            # ---------------- ROC CURVE ----------------
            if hasattr(model, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                fpr, tpr, _ = roc_curve(y_test, y_prob)

                st.subheader("📉 ROC Curve")
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                ax_roc.plot([0, 1], [0, 1], "k--")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend()
                st.pyplot(fig_roc)

        # --------------------------------------------------
        # FUTURE PREDICTION
        # --------------------------------------------------
        st.subheader("🚀 Future Prediction")

        future_file = st.file_uploader(
            "Upload Future Data (CSV / Excel)",
            type=["csv", "xlsx"],
            key="future"
        )

        if future_file:
            if future_file.name.endswith(".csv"):
                future_df = pd.read_csv(future_file)
            else:
                future_df = pd.read_excel(future_file)

            preds = pipeline.predict(future_df[features])
            future_df["Prediction"] = preds

            # Decode labels if needed
            if le is not None:
                future_df["Prediction_Label"] = le.inverse_transform(preds)

            st.dataframe(future_df)

            csv = future_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Prediction File",
                csv,
                "final_predictions.csv",
                "text/csv"
            )

else:
    st.info("Upload a dataset to start analysis.")
