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

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))

y_pred_svr = regressor.predict(x_grid)
y_pred_knn = knn_reg.predict(x_grid)


plt.scatter(x, y, color='red', label='Actual Data')

plt.plot(x_grid, y_pred_svr, color='blue', label='SVR (Polynomial)')

plt.plot(x_grid, y_pred_knn, color='green', label='KNN Regression')

plt.scatter(6.5, knn_reg.predict([[6.5]]), color='black', s=100, label='Prediction @ 6.5')

plt.title('Salary Prediction using SVR & KNN')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)

plt.show()





