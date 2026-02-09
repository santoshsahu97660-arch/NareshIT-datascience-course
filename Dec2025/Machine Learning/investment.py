import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

dataset = pd.read_csv(r"C:\Users\santo\OneDrive\Desktop\Data science\Dec2025\15th, 16th - SLR\Investment.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

x = pd.get_dummies(x, dtype=int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

x = np.append(x, np.full((50, 1), 42467), axis=1).astype(int)

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
33
import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

bias_score = regressor.score(x_train, y_train)
print("Training Score (Bias):", bias_score)

variance_score = regressor.score(x_test, y_test)
print("Testing Score (Variance):", variance_score)
