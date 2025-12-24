import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\santo\OneDrive\Desktop\Data science\Dec2025\23rd Dec 2025\emp_sal.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.svm import SVR
regressor = SVR(kernel= "poly", degree=4, gamma='auto', C=5.0)
regressor.fit(x, y)

from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=4, weights='distance', p=1)
knn_reg.fit(x, y)

y_pred_knn = knn_reg.predict([[6.5]])
print(y_pred_knn)





