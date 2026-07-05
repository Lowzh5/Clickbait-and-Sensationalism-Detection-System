import os
import math
import joblib
from shared_pipeline import BASE_DIR, clean_text
from utils import assign_severity, get_influential_words, format_result

MODEL_PATH = os.path.join(BASE_DIR, "models", "svm.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# load once when this module is imported, so predict_svm() can be called many times cheaply
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def sigmoid(x):
    # squashes decision_function's raw distance-from-margin value into a 0-1 range
    return 1 / (1 + math.exp(-x))

def predict_svm(headline):
    # 1. clean the input headline
    cleaned = clean_text(headline)

    # 2. transform into TF-IDF features using the already-fitted vectorizer
    text_vector = vectorizer.transform([cleaned])

    # 3. LinearSVC has no predict_proba(), so use decision_function() instead.
    # Positive values mean the headline is on the "clickbait" side of the margin, vice versa
    decision_value = model.decision_function(text_vector)[0]

    # 4. convert the raw decision value into a confidence-like probability (0-1),
    # then into a 0-100 clickbait score
    confidence = sigmoid(decision_value)
    clickbait_score = int(round(confidence * 100))

    prediction = "Clickbait" if decision_value > 0 else "Non-clickbait"
    severity = assign_severity(clickbait_score)
    influential_words = get_influential_words(model, vectorizer, text_vector)

    return format_result("SVM", prediction, clickbait_score, severity, influential_words)

if __name__ == "__main__":
    sample_headline = "You Won't Believe"
    result = predict_svm(sample_headline)
    print(result)
