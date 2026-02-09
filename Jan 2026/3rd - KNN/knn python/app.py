import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# Set page configuration
st.set_page_config(
    page_title="KNN Classifier Playground",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #165b8c;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Sidebar for user inputs
st.sidebar.header("⚙️ Model Hyperparameters")

def sidebar_input_features():
    n_neighbors = st.sidebar.slider("Number of Neighbors (K)", min_value=1, max_value=20, value=5)
    weights = st.sidebar.selectbox("Weights", ("uniform", "distance"))
    algorithm = st.sidebar.selectbox("Algorithm", ("auto", "ball_tree", "kd_tree", "brute"))
    return n_neighbors, weights, algorithm

n_neighbors, weights, algorithm = sidebar_input_features()

# Main Header
st.title("🌸 K-Nearest Neighbors (KNN) Classifier")
st.markdown("### Explore the Iris Dataset with KNN")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/320px-Kosaciec_szczecinkowaty_Iris_setosa.jpg", caption="Iris Flower", use_column_width=True)
    st.info("The K-Nearest Neighbors (KNN) algorithm is a simple, supervised machine learning algorithm that can be used to solve both classification and regression problems.")

with col2:
    st.write("### Dataset Preview")
    st.dataframe(df.head(6), height=200)

# 2. Model Training
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.divider()

# 3. Model Performance
st.subheader("📊 Model Performance")
st.metric(label="Accuracy Score", value=f"{acc:.2%}")

# 4. Visualization (PCA to 2D)
st.subheader("🗺️ Decision Boundary Visualization (PCA Reduced)")
st.caption("Reducing the 4D Iris dataset to 2D using PCA to visualize decision boundaries.")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Train a new model on the PCA-reduced data for decision boundary visualization
clf_pca = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
clf_pca.fit(X_pca, y)

# Create a mesh grid
h = .02  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict for each point in the mesh
Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['#FF0000', '#00FF00', '#0000FF']

ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=target_names[y], palette=cmap_bold, alpha=1.0, edgecolor="black", ax=ax)
ax.set_title(f"KNN Decision Boundaries (K={n_neighbors})")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
st.pyplot(fig)

st.divider()

# 5. Prediction Interface
st.subheader("🔍 Make a Prediction")
st.markdown("Enter the flower's measurements to predict its species:")

c1, c2, c3, c4 = st.columns(4)
with c1:
    sepal_len = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
with c2:
    sepal_wid = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
with c3:
    petal_len = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
with c4:
    petal_wid = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict Species"):
    input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    prediction = clf.predict(input_data)
    pred_species = target_names[prediction[0]]
    
    st.success(f"The model predicts: **{pred_species}**")
    
    # Show probability
    proba = clf.predict_proba(input_data)
    proba_df = pd.DataFrame(proba, columns=target_names, index=['Probability'])
    st.bar_chart(proba_df.T)
