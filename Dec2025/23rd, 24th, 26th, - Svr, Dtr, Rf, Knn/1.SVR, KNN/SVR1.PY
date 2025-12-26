# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\kdata\Desktop\KODI WORK\1. NARESH\2. EVENING BATCH\N_Batch -- 6.00PM\3. JUNE\14th\EMP SAL.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=4, algorithm = 'kd_tree')
regressor.fit(X, y)
'''

# Fitting SVR to the dataset
from sklearn.svm import SVR
#imported the svr class from SKLEARN.SVM library
regressor = SVR(kernel = 'poly', degree=5)

#create regressor object & for now understand kernal is use for linear,polynomial or non-linear svr which we will see indepth concept SVM 
#Non-linear data we will use kernal & rbf is the value which we will assign rbf for kernal
regressor.fit(X, y)
#as usual fit the svr regressor 

# Predicting a new result
y_pred = regressor.predict([[6.5]])

#we are checking employee previous salary we found as 130k but actually his sal was 160k which is incorrect
#that is not good at all, we


# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#why wrong prediction happened hear & why we got the blue straight line we got hear
#lets look at this & on this cases we have to do the feature scaling 
#lets see the code with feature scaling technique




