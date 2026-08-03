"""
Ablation study for LinearSVC's C (regularization strength) parameter.
TF-IDF config and train/test split are held fixed at the production values
(ngram_range=(1,2), max_features=50000, random_state=42) - only C changes -
so the results isolate C's effect on accuracy/precision/recall/F1.

Fits everything in-memory - does NOT touch models/*.pkl, so it never
overwrites the vectorizer/model used by predict_svm.py and the Streamlit app.
"""
import os
import sys
import csv

from sklearn.svm import LinearSVC

# data_pipeline.py / evaluation.py live in src/, not under the exploration folder
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_pipeline import BASE_DIR, load_dataset, clean_dataset, tfidf
from evaluation import evaluate_model

C_VALUES = [0.01, 0.1, 1, 10, 100]

OUTPUT_PATH = os.path.join(BASE_DIR, "exploration", "svm_c_ablation_results.csv")

if __name__ == "__main__":
    df = load_dataset()
    cleaned_df = clean_dataset(df)

    # same TF-IDF vectorizer + same train/test split as train_svm.py, fit once and reused for every C
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = tfidf(cleaned_df)

    rows = []
    for c in C_VALUES:
        print(f"\n=== C={c} ===")

        model = LinearSVC(C=c, random_state=42)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        metrics = evaluate_model(y_test, y_pred, model_name=f"SVM (C={c})")

        rows.append({
            "C": c,
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
        })

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUTPUT_PATH}")
