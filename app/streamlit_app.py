import os
import re
import sys

import altair as alt
import pandas as pd
import streamlit as st

# --- make src/ importable -----------------------------------------------
# src modules use flat imports (e.g. "from data_pipeline import ..."),
# so src/ itself needs to be on sys.path rather than imported as a package.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predict_svm import predict_svm
from predict_logistic_regression import predict_logistic_regression
from predict_naive_bayes import predict_naive_bayes 

# --- model registry --------------------------------------------------------
# Maps each selectable model name to its predict function.
MODEL_REGISTRY = {
    "SVM": predict_svm,
    "Naive Bayes": predict_naive_bayes,
    "Logistic Regression": predict_logistic_regression,
    "All Models": None,  # special-cased below - runs all three models and shows them side by side
}

ALL_MODEL_NAMES = ["SVM", "Logistic Regression", "Naive Bayes"]

EXAMPLE_HEADLINES = [
    "You Won't Believe What Happened Next",
    "Government Announces New Education Policy",
    "This Simple Trick Can Change Your Life",
]

MIN_HEADLINE_LENGTH = 10  # characters; used for the "too short" input check

# --- page setup --------------------------------------------------------
st.set_page_config(
    page_title="Clickbait Detection System",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Clickbait Detection System")
st.write(
    "Enter an English news headline below and a trained NLP model will predict "
    "whether it is **Clickbait** or **Non-clickbait**, along with a score, "
    "severity level, and the words that most influenced the decision."
)
# --- input section --------------------------------------------------------
st.subheader("1. Enter a Headline")

st.text_area(
    "Headline text",
    key="headline_input",
    height=80,
    placeholder="Type your own English headline here...",
    label_visibility="collapsed",
)

# Shown purely as inspiration - the user always types their own headline;
# these are not clickable/auto-fill buttons.
st.caption("Need inspiration? Try typing something like:")
for example in EXAMPLE_HEADLINES:
    st.caption(f"- {example}")

# --- model selection section ---------------------------------------------
st.subheader("2. Select a Model")

model_names = list(MODEL_REGISTRY.keys())

selected_model = st.radio(
    "Model",
    options=model_names,
    horizontal=True,
    label_visibility="collapsed",
)

# --- analyze button --------------------------------------------------------
st.subheader("3. Analyze")
analyze_clicked = st.button("🔍 Analyze Headline", type="primary")


def highlight_headline(headline: str, words: list[str]) -> str:
    """
    Return the original headline with each influential word/phrase wrapped
    in markdown highlighting, matched case-insensitively against the raw text.
    """
    if not words:
        return headline

    def word_to_pattern(word: str) -> str:
        sub_patterns = ["'?".join(re.escape(ch) for ch in part) for part in word.split(" ")]
        return r"\s+".join(sub_patterns)

    # longest phrases first so a bigram match isn't shadowed by its own unigram
    sorted_words = sorted(set(words), key=len, reverse=True)
    combined_pattern = "|".join(f"({word_to_pattern(w)})" for w in sorted_words)

    return re.sub(
        combined_pattern,
        lambda m: f":red[**{m.group(0)}**]",
        headline,
        flags=re.IGNORECASE,
    )


def build_contribution_chart(words: list[str], scores: list[float], compact: bool = False) -> alt.Chart:
    """
    Horizontal bar chart of each influential word's contribution score.
    Positive = pushed the prediction towards Clickbait, negative = away from it.
    `compact` drops the legend/axis title for the narrow All-Models columns.
    """
    df = pd.DataFrame({"word": words, "contribution": scores})
    df["direction"] = df["contribution"].apply(
        lambda v: "Toward Clickbait" if v >= 0 else "Away from Clickbait"
    )

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title=None if compact else "Contribution to prediction"),
            y=alt.Y("word:N", sort="-x", title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Toward Clickbait", "Away from Clickbait"],
                    range=["#2a78d6", "#eb6834"],
                ),
                legend=None if compact else alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("word:N", title="Word"),
                alt.Tooltip("contribution:Q", title="Contribution", format=".4f"),
            ],
        )
        .properties(height=alt.Step(22 if compact else 28))
    )


def display_result(result: dict, headline: str) -> None:
    """
    Render the prediction result as cards. `result` follows the shape
    produced by utils.format_result(): model_name, prediction, clickbait_score,
    severity, influential_words, influential_word_scores.
    """
    prediction = result["prediction"]
    score = result["clickbait_score"]
    severity = result["severity"]
    words = result["influential_words"]
    word_scores = result["influential_word_scores"]

    severity_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(severity, "gray")
    prediction_icon = "🚩" if prediction == "Clickbait" else "✅"

    st.divider()
    st.subheader("Result")

    card_cols = st.columns(3)
    with card_cols[0]:
        st.metric("Prediction", f"{prediction_icon} {prediction}")
    with card_cols[1]:
        st.metric("Clickbait Score", f"{score} / 100")
        st.progress(score / 100)
    with card_cols[2]:
        st.markdown("**Severity**")
        st.markdown(f":{severity_color}[**{severity}**]")

    st.markdown("**Model-based Explanation**")
    if words:
        st.markdown(highlight_headline(headline, words))
        st.altair_chart(build_contribution_chart(words, word_scores), use_container_width=True)
    else:
        st.caption("No influential words were found in this headline.")

    st.caption(f"Model used: **{result['model_name']}**")
    st.info(
        "ℹ️ The score, severity, and explanation above are generated directly "
        "from the selected trained model's learned weights on this headline - "
        "they are not based on a manually written trigger-word list."
    )


def display_all_models_result(results: list[dict], headline: str) -> None:
    """Render one compact card per model, side by side, for the "All Models" option."""
    severity_color = {"Low": "green", "Medium": "orange", "High": "red"}

    st.divider()
    st.subheader("Result - All Models")

    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        with col:
            prediction = result["prediction"]
            score = result["clickbait_score"]
            severity = result["severity"]
            words = result["influential_words"]
            word_scores = result["influential_word_scores"]
            prediction_icon = "🚩" if prediction == "Clickbait" else "✅"

            st.markdown(f"**{result['model_name']}**")
            st.metric("Prediction", f"{prediction_icon} {prediction}")
            st.metric("Score", f"{score} / 100")
            st.progress(score / 100)
            st.markdown(f"Severity: :{severity_color.get(severity, 'gray')}[**{severity}**]")

            st.caption("Influential words:")
            if words:
                st.markdown(highlight_headline(headline, words))
                st.altair_chart(build_contribution_chart(words, word_scores, compact=True), use_container_width=True)
            else:
                st.caption("None found.")

    st.info(
        "ℹ️ Each model was trained independently and may disagree - compare "
        "the predictions, scores, and influential words above."
    )


if analyze_clicked:
    headline = st.session_state["headline_input"].strip()

    if not headline:
        st.warning("⚠️ Please enter a headline before analyzing.")
    elif len(headline) < MIN_HEADLINE_LENGTH:
        st.warning(
            "⚠️ That headline looks too short. Please enter a more complete, "
            "meaningful headline for an accurate prediction."
        )
    elif selected_model == "All Models":
        with st.spinner("Analyzing headline with all models..."):
            results = [MODEL_REGISTRY[name](headline) for name in ALL_MODEL_NAMES]
        display_all_models_result(results, headline)
    else:
        predict_fn = MODEL_REGISTRY[selected_model]
        with st.spinner(f"Analyzing headline with {selected_model}..."):
            result = predict_fn(headline)
        display_result(result, headline)