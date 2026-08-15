"""
Generate side-by-side confusion matrix heatmaps for each model using the
shared overall-best ablation configuration (ngram_range=(1,2), stop_words=None,
max_features=50000 -- see Section 4.1.3), to support the Best Model Selection
discussion (Section 4.2.5). All three models are trained on the exact same
TF-IDF features for a fair, apples-to-apples comparison.

This shared config is identical to the production pipeline in
data_pipeline.tfidf(), so models/tfidf_vectorizer.pkl, logistic_regression.pkl,
and svm.pkl are reused directly. Naive Bayes' saved model also happens to use
this same config, but is refit here explicitly for clarity/consistency.
"""
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_pipeline import BASE_DIR, load_dataset, clean_dataset

df = load_dataset()
cleaned_df = clean_dataset(df)
X = cleaned_df["cleaned_headline"]
y = cleaned_df["clickbait"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- shared config (1,2) / None / 50000 for all three models ---------------
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
X_test_shared = vectorizer.transform(X_test)

nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(vectorizer.transform(X_train), y_train)
nb_pred = nb_model.predict(X_test_shared)

lr_model = joblib.load(os.path.join(BASE_DIR, "models", "logistic_regression.pkl"))
svm_model = joblib.load(os.path.join(BASE_DIR, "models", "svm.pkl"))

lr_pred = lr_model.predict(X_test_shared)
svm_pred = svm_model.predict(X_test_shared)

models = {
    "Naive Bayes": (nb_pred, "#3B5A73"),
    "Logistic Regression": (lr_pred, "#9A6A3C"),
    "SVM": (svm_pred, "#6B5B8C"),
}

LABELS = ["Non-clickbait", "Clickbait"]

fig, axes = plt.subplots(1, 3, figsize=(19, 5.3), dpi=150, gridspec_kw={"wspace": 0.35})

for ax, (name, (pred, color)) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, pred)  # rows = actual [0,1], cols = predicted [0,1]
    # start from a light gray (not pure white) so low-count cells (FP/FN) stay
    # visible against the figure's white background
    cmap = LinearSegmentedColormap.from_list("model_cmap", ["#e3e3e3", color])

    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1])
    ax.set_yticks([])  # replaced by the colored row-label bar on the left
    ax.set_xticklabels(LABELS, fontsize=13, rotation=0)
    ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
    ax.set_title(name, fontsize=15, fontweight="bold")

    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     fontsize=16, color=text_color, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # row-label strip to the left of the matrix, with text rotated 90deg
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="10%", pad=0.05)
    cax.set_xlim(0, 1)
    cax.set_ylim(-0.5, 1.5)
    cax.invert_yaxis()  # row 0 (Non-clickbait) at top, matching imshow's row order
    for row_idx, label in enumerate(LABELS):
        cax.text(0.5, row_idx, label, rotation=90, ha="center", va="center", fontsize=13)
    cax.set_xticks([])
    cax.set_yticks([])
    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.text(-0.9, 0.5, "Actual", rotation=90, ha="center", va="center",
              fontsize=13, transform=cax.transAxes)

fig.suptitle("Confusion Matrices for Each Model", fontsize=17, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.99])

output_path = os.path.join(BASE_DIR, "reports", "Image", "confusion_matrix_best_configs.png")
fig.savefig(output_path, bbox_inches="tight")
print(f"Saved to {output_path}")
