import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\santo\OneDrive\Desktop\Data science\Dec2025\23rd Dec 2025\emp_sal.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion = 'friedman_mse',splitter = 'random')   
regressor.fit(X, y)

from sklearn.ensemble import RandomForestRegressor 
reg = RandomForestRegressor(n_estimators = 300, random_state = 0)
reg.fit(X,y)

y_pred = reg.predict([[6.5]])
print(y_pred)

plt.scatter(X, y, color = 'red')
plt.plot(X,regressor.predict(X), color = 'blue')
plt.title('Truth or bluff (Decision tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()