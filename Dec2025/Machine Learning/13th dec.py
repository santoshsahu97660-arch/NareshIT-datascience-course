import numpy as np

import matplotlib.pyplot as plt

import pandas as pd 

df = pd.read_csv(r"C:\Users\santo\Downloads\Data.csv")

print(df.head())

x = df.iloc[:, :-1].values

y = df.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

imputer = imputer.fit(x[:,1:3])

x[:,1:3] = imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()

x[:,0] = labelencoder_x.fit_transform(x[:,0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.8, random_state=0)

