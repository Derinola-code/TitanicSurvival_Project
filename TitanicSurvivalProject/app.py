from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model/titanic_survival_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    fare = float(request.form['fare'])
    embarked = int(request.form['embarked'])

    data = np.array([[pclass, sex, age, fare, embarked]])
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    result = "Survived" if prediction[0] == 1 else "Did Not Survive"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
