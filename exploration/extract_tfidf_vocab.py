"""
Dumps the current models/tfidf_vectorizer.pkl vocabulary and IDF scores to text
files for manual inspection. Depends only on the vectorizer (not on any specific
trained model), so re-run this after every ablation study (e.g. changing
ngram_range, stop_words, max_features in data_pipeline.tfidf()) to refresh
tfidf_features_list.txt / tfidf_idf_scores.txt against the latest vectorizer.
"""
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
feature_names = vectorizer.get_feature_names_out()
idf_scores = vectorizer.idf_

features_path = os.path.join(BASE_DIR, "exploration", "tfidf_features_list.txt")
with open(features_path, "w", encoding="utf-8") as f:
    for name in feature_names:
        f.write(f"{name}\n")
print(f"Saved {len(feature_names)} features to {features_path}")

idf_path = os.path.join(BASE_DIR, "exploration", "tfidf_idf_scores.txt")
with open(idf_path, "w", encoding="utf-8") as f:
    f.write(f"{'Feature':<30} | {'IDF Score'}\n")
    f.write("-" * 50 + "\n")
    for name, score in zip(feature_names, idf_scores):
        f.write(f"{name:<30} | {score:.4f}\n")
print(f"Saved {len(feature_names)} IDF scores to {idf_path}")
