from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

app = Flask(__name__)

# Load the trained model and vectorizer
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# API Route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(text_tfidf)
    prediction_proba = model.predict_proba(text_tfidf)[:, 1]  # Probability for class 1 (Positive)
    sentiment_label = "Positive" if prediction[0] == 1 else "Negative"
    return jsonify({"sentiment": sentiment_label, "sentiment_score": float(prediction_proba[0]), "text": text})

# Web UI Route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        processed_text = preprocess_text(text)
        text_tfidf = vectorizer.transform([processed_text])
        prediction = model.predict(text_tfidf)
        prediction_proba = model.predict_proba(text_tfidf)[:, 1]
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        return render_template("index.html", text=text, sentiment=sentiment, sentiment_score=round(float(prediction_proba[0]), 2))
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
