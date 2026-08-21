"""
Ablation study for TfidfVectorizer(ngram_range, stop_words, max_features).
Trains Naive Bayes, Logistic Regression, and SVM under each of the 18
combinations and records accuracy/precision/recall/F1 for every (config, model)
pair into exploration/ablation_results.csv.
"""
import os
import sys
import csv
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# data_pipeline.py / evaluation.py live in src/, not under the exploration folder
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_pipeline import BASE_DIR, load_dataset, clean_dataset
from evaluation import evaluate_model

NGRAM_RANGES = [(1, 1), (1, 2), (1, 3)]
STOP_WORDS_OPTIONS = [None, "english"]
MAX_FEATURES_OPTIONS = [None, 50000, 100000]

OUTPUT_PATH = os.path.join(BASE_DIR, "exploration", "ablation_results.csv")

def build_models():
    return {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "SVM": LinearSVC(C=1.0, random_state=42),
    }

if __name__ == "__main__":
    df = load_dataset()
    cleaned_df = clean_dataset(df)

    X = cleaned_df["cleaned_headline"]
    y = cleaned_df["clickbait"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rows = []
    combos = list(itertools.product(NGRAM_RANGES, STOP_WORDS_OPTIONS, MAX_FEATURES_OPTIONS))

    for combo_num, (ngram_range, stop_words, max_features) in enumerate(combos, start=1):
        print(f"\n=== Combo {combo_num}/{len(combos)}: ngram_range={ngram_range}, "
              f"stop_words={stop_words}, max_features={max_features} ===")

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words,
            max_features=max_features,
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        for model_name, model in build_models().items():
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            metrics = evaluate_model(y_test, y_pred, model_name=model_name)

            rows.append({
                "ngram_range": f"{ngram_range[0]} to {ngram_range[1]}",
                "stop_words": "enabled" if stop_words else "disabled",
                "max_features": max_features if max_features else "None",
                "model": model_name,
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
            })

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows ({len(combos)} configs x {len(build_models())} models) to {OUTPUT_PATH}")
