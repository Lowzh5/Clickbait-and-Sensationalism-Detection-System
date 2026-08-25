import os
import joblib
from data_pipeline import BASE_DIR, clean_text
from inference_utils import assign_severity, get_influential_words, format_result

MODEL_PATH = os.path.join(BASE_DIR, "models", "naive_bayes.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

"""
Loads the saved Naive Bayes model 
and predicts whether a headline is clickbait.
"""
def predict_naive_bayes(headline):
    # Clean headline
    cleaned = clean_text(headline)

    # Transform into TFIDF features
    text_vector = vectorizer.transform([cleaned])

    # Prediction
    prob = model.predict_proba(text_vector)[0]
    clickbait_prob = prob[1]
    clickbait_score = int(round(clickbait_prob * 100))

    prediction = "Clickbait" if clickbait_score > 50 else "Non-clickbait"
    severity = assign_severity(clickbait_score)

    # Extract influential words
    influential_words, influential_scores = get_influential_words(model, vectorizer, text_vector)

    return format_result("Naive Bayes", prediction, clickbait_score, severity, influential_words, influential_scores)

if __name__ == "__main__":
    sample_headline = "You Won't Believe This Shocking Secret!"
    result = predict_naive_bayes(sample_headline)
    print(result)