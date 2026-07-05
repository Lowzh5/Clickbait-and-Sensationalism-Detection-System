import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "svm.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))

feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]

sorted_idx = np.argsort(weights)[::-1]

output_path = os.path.join(BASE_DIR, "models", "svm_weights.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"{'Word':<30} {'Weight':>10}\n")
    f.write("-" * 42 + "\n")
    for idx in sorted_idx:
        f.write(f"{feature_names[idx]:<30} {weights[idx]:>10.6f}\n")

print(f"Saved {len(feature_names)} weights to {output_path}")
