from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("music_model.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        gender = int(request.form["gender"])

        if age <= 0:
            return render_template(
                "index.html",
                prediction_text="Error: Age must be a positive value."
            )

        input_data = pd.DataFrame(
            [[age, gender]],
            columns=["age", "gender"]
        )

        prediction = model.predict(input_data)
        predicted_genre = genre_encoder.inverse_transform(prediction)[0]

        return render_template(
            "index.html",
            prediction_text="Recommended Music Genre: " + predicted_genre
        )

    except Exception:
        return render_template(
            "index.html",
            prediction_text="Error: Please enter valid input values."
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)

