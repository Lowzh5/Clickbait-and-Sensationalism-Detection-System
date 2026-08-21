"""
Generates the per-model top-10 influential words chart used in Discussion
4.2.4 (Influential Words Interpretation): the 10 highest-weighted words/
n-grams learned by each trained model (Naive Bayes, Logistic Regression, SVM).
"""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
feature_names = np.array(vectorizer.get_feature_names_out())

nb = joblib.load(os.path.join(MODELS_DIR, "naive_bayes.pkl"))
lr = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
svm = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))

# NB has no coef_; feature_log_prob_[1] - feature_log_prob_[0] is its equivalent
# clickbait-leaning weight per word (same logic as inference_utils.get_influential_words)
model_weights = {
    "Naive Bayes": (nb.feature_log_prob_[1] - nb.feature_log_prob_[0], "#3B5A73"),
    "Logistic Regression": (lr.coef_[0], "#9A6A3C"),
    "SVM": (svm.coef_[0], "#6B5B8C"),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=150)

for ax, (model_name, (weights, color)) in zip(axes, model_weights.items()):
    top_idx = np.argsort(weights)[::-1][:10]
    words = feature_names[top_idx][::-1]  # first-ranked word at top
    scores = weights[top_idx][::-1]
    y = range(len(words))

    ax.barh(list(y), scores, height=0.6, color=color)

    for i, s in enumerate(scores):
        ax.text(s, i, f" {s:.2f}", va="center", ha="left", fontsize=8.5, color="#333333")

    ax.set_yticks(list(y))
    ax.set_yticklabels(words)
    ax.set_xlim(0, max(scores) * 1.18)
    ax.set_xlabel("Model weight")
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

fig.suptitle("Top 10 Most Influential Words/Phrases Learned by Each Model", fontsize=15, y=1.03)
fig.tight_layout()

output_path = os.path.join(BASE_DIR, "reports", "Image", "top_influential_words_by_model.png")
fig.savefig(output_path, bbox_inches="tight")
print(f"Saved to {output_path}")