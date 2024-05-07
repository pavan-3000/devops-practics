from flask import Flask,render_template,request
import joblib
import numpy as np

model = joblib.load("model.pkl")

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_f = [int(x) for x in request.form.values()]
        final_f = np.array(input_f).reshape(1, -1)  # Reshape the input data

        prediction = model.predict(final_f)
        return render_template("result.html", prediction=prediction)
    else:
        # Handle other HTTP methods (e.g., GET) if needed
        return "Method Not Allowed"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3000)
