import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Sample dataset (experience vs salary)
data = {
    'experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'salary': [15000, 20000, 25000, 30000, 35000,
               40000, 45000, 50000, 55000, 60000]
}

df = pd.DataFrame(data)

X = df[['experience']]
y = df['salary']

model = LinearRegression()
model.fit(X, y)

# Save model
with open('salary_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained & saved as salary_model.pkl")
