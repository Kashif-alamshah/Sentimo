from flask import Flask, render_template, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from googletrans import Translator

app = Flask(__name__)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

translator = Translator()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    user_text = request.form.get("text")
    language = request.form.get("language")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    if language != "en":
        translated = translator.translate(user_text, src=language, dest="en")
        user_text = translated.text

    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_score = probs[0][1].item() - probs[0][0].item()
    scaled_score = round(sentiment_score, 2)
    return jsonify({"sentiment": scaled_score})

@app.route("/about")
def about():
    return render_template("about.html", title="About - Sentimo")

@app.route("/data")
def data():
    return render_template("data.html", title="Data - Sentimo")

if __name__ == "__main__":
    app.run(debug=True)