"""
Generates the top-words-by-class chart used in Methodology 3.2 (preliminary
data analysis): The 10 words most frequent in clickbait headlines 
and non-clickbait headlines.
"""
import os
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "cleaned_clickbait_data.csv"))

STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "is", "are",
        "this", "that", "with", "from", "at", "by"}

def top_words(subset, n=10):
    words = " ".join(subset["cleaned_headline"]).split()
    words = [w for w in words if w not in STOP and len(w) > 2]
    return [w for w, _ in Counter(words).most_common(n)]

def count_containing(subset, word):
    pattern = r"\b" + re.escape(word) + r"\b"
    return subset["cleaned_headline"].str.contains(pattern, regex=True).sum()

non_clickbait_df = df[df["clickbait"] == 0]
clickbait_df = df[df["clickbait"] == 1]

clickbait_top = top_words(clickbait_df)
non_clickbait_top = top_words(non_clickbait_df)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=150)

panels = [
    (axes[0], clickbait_top, "Words Most Common in Clickbait Headlines"),
    (axes[1], non_clickbait_top, "Words Most Common in Non-clickbait Headlines"),
]

for ax, words, title in panels:
    words = words[::-1]  # first-ranked word at top
    y = range(len(words))

    non_click_share, click_share = [], []
    for w in words:
        n_non = count_containing(non_clickbait_df, w)
        n_click = count_containing(clickbait_df, w)
        total = n_non + n_click
        non_click_share.append(n_non / total * 100)
        click_share.append(n_click / total * 100)

    ax.barh(list(y), non_click_share, height=0.6, color="#3B5A73", label="Non-clickbait")
    ax.barh(list(y), click_share, height=0.6, left=non_click_share, color="#9A6A3C", label="Clickbait")

    for i, (nc, c) in enumerate(zip(non_click_share, click_share)):
        # label anchored to the ends of the full-width bar, never inside a thin segment
        ax.text(2, i, f"{nc:.1f}%", va="center", ha="left", fontsize=9,
                 color="white")
        ax.text(98, i, f"{c:.1f}%", va="center", ha="right", fontsize=9,
                 color="white")

    ax.set_yticks(list(y))
    ax.set_yticklabels(words)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of headlines containing the word (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=9, frameon=False)
fig.suptitle("Word Usage by Headline Class: Clickbait vs Non-Clickbait", fontsize=15, y=1.02)
fig.tight_layout()

output_path = os.path.join(BASE_DIR, "reports", "Image", "top_words_prevalence_by_class.png")
fig.savefig(output_path, bbox_inches="tight")
