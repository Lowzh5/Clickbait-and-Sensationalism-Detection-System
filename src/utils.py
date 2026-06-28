"""
assign_severity(score) - maps the clickbait score to Low / Medium / High (0-30, 31-65, 66-100)
get_influential_words(model, vectorizer, text_vector) - extracts the top influential TF-IDF words from a model's coefficients/weights for a given input
format_result(model_name, prediction, score, severity, words) - builds the standard output dict that all predict functions return

This avoids each member writing the same severity logic and output formatting 
separately in their own predict file. The model-specific part (getting probability, loading 
the right .pkl) stays in each predict file, but the common logic lives in utils.py.

"""