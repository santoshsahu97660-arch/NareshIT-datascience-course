import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv(r"C:\Users\santo\Downloads\Data.csv")

print(df.head())

x = df.iloc[:, :-1].values

y = df.iloc[:,-3].values
