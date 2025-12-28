import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    HuberRegressor
)

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ================== LOAD DATA ==================
data = pd.read_csv(
    r"C:\Users\santo\OneDrive\Desktop\Data science\Dec2025\27 dec 2025\USA_Housing\USA_Housing.csv"
)

X = data.drop(['Price', 'Address'], axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ================== MODELS ==================
models = {
    'LinearRegression': LinearRegression(),
    'RobustRegression': HuberRegressor(),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),

    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('linear', LinearRegression())
    ]),

    'SGDRegressor': SGDRegressor(),
    'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    'RandomForest': RandomForestRegressor(random_state=0),
    'SVM': SVR(),
    'LGBM': lgb.LGBMRegressor(random_state=0),
    'XGBoost': xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=0
    ),
    'KNN': KNeighborsRegressor()
}

# ================== TRAIN + EVALUATE ==================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })

    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# ================== SAVE RESULTS ==================
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

print("All models trained successfully")
print("All models saved as pickle files")
print("Evaluation results saved to model_evaluation_results.csv")
