"""
Generate the chart to show the F1-score performance between stop_word = None vs = English
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv(os.path.join(BASE_DIR, "exploration", "ablation_results.csv"))

models = ["Naive Bayes", "Logistic Regression", "SVM"]
none_f1 = [df[(df.model == m) & (df.stop_words == "disabled")]["f1"].mean() for m in models]
english_f1 = [df[(df.model == m) & (df.stop_words == "enabled")]["f1"].mean() for m in models]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

bars1 = ax.bar(x - width / 2, none_f1, width, label='stop_words=None', color="#3B5A73")
bars2 = ax.bar(x + width / 2, english_f1, width, label='stop_words="english"', color="#9A6A3C")

for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.001, f"{height:.4f}",
                 ha="center", va="bottom", fontsize=9)

ax.set_ylabel("F1-score (averaged across ablation configurations)")
ax.set_title("Effect of Stop-Word Removal on F1-score by Model", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.93, 0.995)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

fig.tight_layout()

output_path = os.path.join(BASE_DIR, "reports", "Image", "f1_stopword_comparison.png")
fig.savefig(output_path, bbox_inches="tight")
print(f"Saved to {output_path}")
