import os
import joblib
from data_pipeline import BASE_DIR, clean_text
from inference_utils import assign_severity, get_influential_words, format_result

MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# load once when this module is imported, so predict_logistic_regression() can be called many times cheaply
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_logistic_regression(headline):
    # 1. clean the input headline
    cleaned = clean_text(headline)

    # 2. transform into TF-IDF features using the already-fitted vectorizer
    text_vector = vectorizer.transform([cleaned])

    # 3. LogisticRegression has predict_proba() built in, unlike LinearSVC,
    # so no manual sigmoid conversion is needed here.
    # predict_proba returns [[P(non-clickbait), P(clickbait)]], we want the clickbait column
    confidence = model.predict_proba(text_vector)[0][1]

    clickbait_score = int(round(confidence * 100))

    prediction = "Clickbait" if confidence > 0.5 else "Non-clickbait"
    severity = assign_severity(clickbait_score)
    influential_words = get_influential_words(model, vectorizer, text_vector)

    return format_result("Logistic Regression", prediction, clickbait_score, severity, influential_words)

if __name__ == "__main__":
    sample_headline = "You Won't Believe"
    result = predict_logistic_regression(sample_headline)
    print(result)