from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# ================= LOAD MODELS =================
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression',
    'LassoRegression', 'ElasticNet', 'PolynomialRegression',
    'SGDRegressor', 'ANN', 'RandomForest', 'SVM',
    'LGBM', 'XGBoost', 'KNN'
]

models = {name: pickle.load(open(f'{name}.pkl', 'rb')) for name in model_names}

results_df = pd.read_csv('model_evaluation_results.csv')

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']

    input_df = pd.DataFrame([{
        'Avg. Area Income': float(request.form['Avg. Area Income']),
        'Avg. Area House Age': float(request.form['Avg. Area House Age']),
        'Avg. Area Number of Rooms': float(request.form['Avg. Area Number of Rooms']),
        'Avg. Area Number of Bedrooms': float(request.form['Avg. Area Number of Bedrooms']),
        'Area Population': float(request.form['Area Population'])
    }])

    model = models.get(model_name)
    prediction = model.predict(input_df)[0]

    return render_template(
        'results.html',
        prediction=round(prediction, 2),
        model_name=model_name
    )

@app.route('/results')
def results():
    return render_template(
        'model.html',
        tables=[results_df.to_html(classes='table table-striped')],
        titles=results_df.columns.values
    )

if __name__ == '__main__':
    app.run(debug=True)
