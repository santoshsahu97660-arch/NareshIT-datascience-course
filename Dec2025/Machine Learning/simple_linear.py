import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\santo\OneDrive\Desktop\Machine learning linear regression\simple_linear_regression_dataset.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Model trained successfully ✅")

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, model.predict(x_train), color='blue')
plt.title('Exam_score vs Hours_Studied(Test set)')
plt.xlabel('Hours_Studied')
plt.ylabel('Exam_score')
plt.show()

m_slope = model.coef_
print("Slope (m):", m_slope)

c_intercept = model.intercept_
print("Intercept (c):", c_intercept)

y_12 = m_slope*12+c_intercept
print(y_12)


y_20= m_slope*20+c_intercept
print(y_20)

bias_score = model.score(x_train, y_train)
print("Training Score (Bias):", bias_score)

variance_score = model.score(x_test, y_test)
print("Testing Score (Variance):", variance_score)









