
from flask import Flask, render_template, request
import joblib
import re
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "fake_news_model.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "count_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    original_text = ""

    if request.method == "POST":

        original_text = request.form.get("news", "").strip()

        if not original_text:
            prediction = "Please enter a news statement."

        elif len(original_text) > 5000:
            prediction = "News statement is too long."

        elif not clean_text(original_text):
            prediction = "Please enter meaningful English text."

        else:
            cleaned = clean_text(original_text)
            features = vectorizer.transform([cleaned])
            result = model.predict(features)[0]

            if result == "Fake":
                prediction = "FAKE"
            else:
                prediction = "REAL"

    return render_template(
        "index.html",
        prediction=prediction,
        news=original_text
    )


if __name__ == "__main__":
    app.run(debug=True)
