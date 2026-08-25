import numpy as np

"""
assign_severity(score) - maps the clickbait score to Low / Medium / High (0-33, 34-66, 67-100)
get_influential_words(model, vectorizer, text_vector) - extracts the top influential TF-IDF words from a model's coefficients/weights for a given input to show in Streamlit
format_result(model_name, prediction, score, severity, words) - builds the standard output dict that all predict functions return
"""
def assign_severity(score):
    # score is expected to be an int only after round up
    if score <= 33:
        return "Low"
    elif score <= 66:
        return "Medium"
    else:
        return "High"

def get_influential_words(model, vectorizer, text_vector, top_n=5):
    # Obtain the vocabulary list to understand which word corresponds to each column in the matrix.
    feature_names = np.array(vectorizer.get_feature_names_out())

    # coef_[0] because this is binary classification (1 row of weights)
    if hasattr(model, "coef_"):
        # use for logistic regression and also the SVM
        # coef_ shape is (1,140121), coef_[0] is take the 'row 0'
        weights = model.coef_[0]
    else:
        # [0] = non-clickbait , [1] clickbait
        # example: 'you' in [0] is -10.68 and in [1] is -5.21, diff is 5.46
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]

    # text_vector is a sparse row, toarray() then flatten to a plain 1D array
    tfidf_values = text_vector.toarray().flatten()

    # only look at words that actually appear in this headline
    present_idx = np.where(tfidf_values > 0)[0]

    # weights is the weight inside the decision_function()
    # the diff is contribution are store the TFIDF * weight for EACH word
    contributions = tfidf_values[present_idx] * weights[present_idx]

    # sort words (and their scores, same order) by how strongly they push towards clickbait
    order = np.argsort(contributions)[::-1]
    ranked_idx = present_idx[order]
    sorted_contributions = contributions[order]

    top_words = feature_names[ranked_idx[:top_n]].tolist()
    top_scores = sorted_contributions[:top_n].tolist()
    return top_words, top_scores

def format_result(model_name, prediction, score, severity, words, word_scores):
    return {
        "model_name": model_name,
        "prediction": prediction,
        "clickbait_score": score,
        "severity": severity,
        "influential_words": words,
        "influential_word_scores": word_scores,
    }
