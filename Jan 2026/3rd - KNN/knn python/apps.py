import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="KNN Classifier Playground",
    page_icon="🌸",
    layout="wide"
)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ KNN Hyperparameters")

k = st.sidebar.slider("Number of Neighbors (K)", 1, 20, 5)
weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
algorithm = st.sidebar.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

# ---------------- Main UI ----------------
st.title("🌸 K-Nearest Neighbors (KNN) Classifier")
st.write("Explore the Iris dataset interactively using KNN")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- Train Model ----------------
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(
    n_neighbors=k,
    weights=weights,
    algorithm=algorithm
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{accuracy:.2%}")

# ---------------- PCA Visualization ----------------
st.subheader("🗺️ Decision Boundary (PCA – 2D)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_model = KNeighborsClassifier(
    n_neighbors=k,
    weights=weights,
    algorithm=algorithm
)
pca_model.fit(X_pca, y)

h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = pca_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 6))
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])

ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

labels = [target_names[i] for i in y]
ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap=ListedColormap(["red", "green", "blue"]),
    s=40
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title(f"KNN Decision Boundary (K={k})")

st.pyplot(fig)

# ---------------- Prediction Section ----------------
st.subheader("🔍 Predict Iris Species")

c1, c2, c3, c4 = st.columns(4)

with c1:
    sl = st.number_input("Sepal Length", 0.0, 10.0, 5.0)
with c2:
    sw = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
with c3:
    pl = st.number_input("Petal Length", 0.0, 10.0, 1.4)
with c4:
    pw = st.number_input("Petal Width", 0.0, 10.0, 0.2)

if st.button("Predict"):
    sample = np.array([[sl, sw, pl, pw]])
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]

    st.success(f"Predicted Species: **{target_names[pred]}**")

    prob_df = pd.DataFrame(
        proba,
        index=target_names,
        columns=["Probability"]
    )
    st.bar_chart(prob_df)
