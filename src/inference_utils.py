"""
assign_severity(score) - maps the clickbait score to Low / Medium / High (0-30, 31-65, 66-100)
get_influential_words(model, vectorizer, text_vector) - extracts the top influential TF-IDF words from a model's coefficients/weights for a given input
format_result(model_name, prediction, score, severity, words) - builds the standard output dict that all predict functions return
"""
import numpy as np


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
    Returns (words, scores) - scores can be negative (word pushed away from
    Clickbait) as well as positive, both are kept so callers can visualize direction.
    """
    # Obtain the vocabulary list to understand which word corresponds to each column in the matrix.
    feature_names = np.array(vectorizer.get_feature_names_out())

    # coef_[0] because this is binary classification (1 row of weights)
    # 模型对每个词学到的权重
    if hasattr(model, "coef_"):
        # use for logistic regression and also the SVM
        # coef_ shape is (1,140121), coef_[0] is take the 'row 0'
        weights = model.coef_[0]
    else:
        # use for naive bayes cause not the linear regression model
        # [0] = non-clickbait , [1] clickbait
        # example: 'you' in [0] is -10.68 and in [1] is -5.21, diff is 5.46
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]

    # text_vector is a sparse row, toarray() then flatten to a plain 1D array
    # 这条headline里每个词的tfidf值(大部分是0)
    tfidf_values = text_vector.toarray().flatten()

    # only look at words that actually appear in this headline
    # 只看这条headline里真正出现的词
    present_idx = np.where(tfidf_values > 0)[0]

    # contribution towards the "clickbait" class for each present word
    # weights are same with the weight inside the decision_function()
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
