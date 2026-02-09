import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\santo\OneDrive\Desktop\Data science\Dec2025\23rd Dec 2025\emp_sal.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(
    criterion='poisson',
    max_depth=3,
    splitter="random",
    random_state=0
)

dt_reg.fit(x, y)

dt_pred = dt_reg.predict([[6.5]])
print(dt_pred)

# Random Forest Algorithm
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(
    n_estimators=20,
    criterion="absolute_error",
    max_depth=1,
    min_samples_split=2,
    random_state=43
)

rf_reg.fit(x, y)

rf_pred = rf_reg.predict([[6.5]])
print(rf_pred)

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y)
plt.plot(x_grid, rf_reg.predict(x_grid))
plt.title("Random Forest Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


