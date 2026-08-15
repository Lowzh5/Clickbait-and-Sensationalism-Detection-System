"""
Generate the ROC curve / AUC comparison chart for each model using the shared
overall-best ablation configuration (ngram_range=(1,2), stop_words=None,
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.metrics import roc_curve, auc
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
nb_scores = nb_model.predict_proba(X_test_shared)[:, 1]

lr_model = joblib.load(os.path.join(BASE_DIR, "models", "logistic_regression.pkl"))
svm_model = joblib.load(os.path.join(BASE_DIR, "models", "svm.pkl"))

lr_scores = lr_model.predict_proba(X_test_shared)[:, 1]
svm_scores = svm_model.decision_function(X_test_shared)  # LinearSVC has no predict_proba

# --- ROC curves + AUC -------------------------------------------------------
curves = {
    "Naive Bayes": (nb_scores, "#3B5A73"),
    "Logistic Regression": (lr_scores, "#9A6A3C"),
    "SVM": (svm_scores, "#6B5B8C"),
}

fig, ax = plt.subplots(figsize=(7.5, 7), dpi=150)

roc_data = {}
for label, (scores, color) in curves.items():
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    roc_data[label] = (fpr, tpr, color, roc_auc)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random Guess (AUC = 0.5000)")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves for Each Model", fontsize=13, fontweight="bold")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# zoomed inset on the top-left corner, where the three curves actually differ
ZOOM_XLIM = (0.0, 0.08)
ZOOM_YLIM = (0.90, 1.005)

axins = inset_axes(ax, width="55%", height="55%", loc="lower right",
                    bbox_to_anchor=(0.0, 0.06, 1, 1), bbox_transform=ax.transAxes)
for label, (fpr, tpr, color, roc_auc) in roc_data.items():
    axins.plot(fpr, tpr, color=color, linewidth=2)
axins.set_xlim(*ZOOM_XLIM)
axins.set_ylim(*ZOOM_YLIM)
axins.set_xticks([0.0, 0.04, 0.08])
axins.set_yticks([0.90, 0.95, 1.00])
axins.tick_params(labelsize=8)
axins.set_facecolor("#f7f7f7")
for spine in axins.spines.values():
    spine.set_edgecolor("#888888")

mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="#888888", linewidth=1)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)

fig.tight_layout()

output_path = os.path.join(BASE_DIR, "reports", "Image", "auc_roc_best_configs.png")
fig.savefig(output_path, bbox_inches="tight")
print(f"Saved to {output_path}")
