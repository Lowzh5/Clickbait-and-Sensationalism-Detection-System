"""
assign_severity(score) - maps the clickbait score to Low / Medium / High (0-30, 31-65, 66-100)
get_influential_words(model, vectorizer, text_vector) - extracts the top influential TF-IDF words from a model's coefficients/weights for a given input
format_result(model_name, prediction, score, severity, words) - builds the standard output dict that all predict functions return
evaluate_model(y_test, y_pred, model_name) - prints and returns accuracy, precision, recall, f1, confusion matrix
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

def evaluate_model(y_test, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{model_name} evaluation on test set:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:") # support = how many test set inside the class
    print(classification_report(y_test, y_pred, target_names=["Non-clickbait", "Clickbait"]))

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def assign_severity(score):
    # score is expected to be an int/float between 0 and 100
    if score <= 30:
        return "Low"
    elif score <= 65:
        return "Medium"
    else:
        return "High"


def get_influential_words(model, vectorizer, text_vector, top_n=5):
    """
    Works for linear models that expose coef_ (e.g. LinearSVC, LogisticRegression).
    text_vector is the TF-IDF vector (1 row) for a single headline.
    Contribution of each word = tfidf value of that word * model weight for that word.
    We only keep words that are actually present in the headline (non-zero TF-IDF).
    """
    # Obtain the vocabulary list to understand which word corresponds to each column in the matrix.
    feature_names = np.array(vectorizer.get_feature_names_out())

    # coef_[0] because this is binary classification (1 row of weights)
    weights = model.coef_[0]

    # text_vector is a sparse row, toarray() then flatten to a plain 1D array
    tfidf_values = text_vector.toarray().flatten()

    # only look at words that actually appear in this headline
    present_idx = np.where(tfidf_values > 0)[0]

    # contribution towards the "clickbait" class for each present word
    contributions = tfidf_values[present_idx] * weights[present_idx]

    # sort words by how strongly (positively) they push towards clickbait
    ranked_idx = present_idx[np.argsort(contributions)[::-1]]

    top_words = feature_names[ranked_idx[:top_n]].tolist()
    return top_words


def format_result(model_name, prediction, score, severity, words):
    return {
        "model_name": model_name,
        "prediction": prediction,
        "clickbait_score": score,
        "severity": severity,
        "influential_words": words,
    }
