import logging
import os
from pathlib import Path

from flask import Flask, render_template, request
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = Path(__file__).parent

try:
    model = joblib.load(BASE_DIR / "music_model.pkl")
    genre_encoder = joblib.load(BASE_DIR / "genre_encoder.pkl")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.critical("Failed to load models: %s", e, exc_info=True)
    raise


@app.route("/")
def home() -> str:
    return render_template("index.html")


@app.route("/health")
def health() -> tuple[dict, int]:
    return {"status": "healthy"}, 200


@app.route("/predict", methods=["POST"])
def predict() -> str:
    try:
        age = int(request.form["age"])
        gender = int(request.form["gender"])

        if age <= 0 or age > 120:
            return render_template(
                "index.html",
                prediction_text="Error: Age must be between 1 and 120.",
            )

        if gender not in (0, 1):
            return render_template(
                "index.html",
                prediction_text="Error: Gender must be 0 (Female) or 1 (Male).",
            )

        input_data = pd.DataFrame([[age, gender]], columns=["age", "gender"])
        prediction = model.predict(input_data)
        predicted_genre = genre_encoder.inverse_transform(prediction)[0]

        logger.info("Prediction: age=%s, gender=%s -> %s", age, gender, predicted_genre)

        return render_template(
            "index.html",
            prediction_text=f"Recommended Music Genre: {predicted_genre}",
        )

    except (ValueError, KeyError) as e:
        logger.warning("Invalid prediction input: %s", e)
        return render_template(
            "index.html",
            prediction_text="Error: Please enter valid input values.",
        )
    except Exception as e:
        logger.error("Unexpected error during prediction: %s", e, exc_info=True)
        return render_template(
            "index.html",
            prediction_text="Error: Something went wrong. Please try again.",
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
