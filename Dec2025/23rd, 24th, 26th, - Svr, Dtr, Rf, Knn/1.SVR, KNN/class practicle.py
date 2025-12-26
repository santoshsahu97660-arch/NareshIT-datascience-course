import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Admin\Desktop\NIT\1. NIT_Batches\1. MORNING BATCH\N_Batch -- 10.00AM_ DEC25\3. Aug\25th, 26th - Svr, Dtr, Rf, Knn\emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# svr model
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly', degree = 4, gamma = 'auto', C=10 )
svr_regressor.fit(X,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)

# knn model 
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=5,weights='distance', leaf_size=30 )
knn_reg_model.fit(X,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

# decission tree 
from sklearn.tree import DecisionTreeRegressor
dtr_reg_model = DecisionTreeRegressor(criterion='absolute_error', max_depth=10, splitter='random')
dtr_reg_model.fit(X,y)

dtr_reg_pred = dtr_reg_model.predict([[6.5]])
print(dtr_reg_pred)


# Random forest 
from sklearn.ensemble import RandomForestRegressor
rfr_reg_model = RandomForestRegressor(n_estimators=6, random_state=0)
rfr_reg_model.fit(X,y)

rfr_reg_pred = rfr_reg_model.predict([[6.5]])
print(rfr_reg_pred)












