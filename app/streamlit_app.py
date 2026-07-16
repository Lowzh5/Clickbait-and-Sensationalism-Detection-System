"""
Clickbait Detection System - Streamlit UI

Pipeline: user headline -> text cleaning -> TF-IDF vectorizer -> trained model -> prediction

How to run:
    streamlit run app/streamlit_app.py

How to add a new model later (e.g. once Naive Bayes is trained):
    1. Implement predict_naive_bayes(headline) in src/predict_naive_bayes.py so it
       returns the same dict shape as predict_svm() (see format_result in src/utils.py).
    2. Import it below (next to the predict_svm import).
    3. In MODEL_REGISTRY, set "available": True and "predict_fn": predict_naive_bayes.
    That's it - the rest of the UI (selection, validation, result cards) needs no changes.
"""

import os
import sys

import streamlit as st

# --- make src/ importable -----------------------------------------------
# src modules use flat imports (e.g. "from shared_pipeline import ..."),
# so src/ itself needs to be on sys.path rather than imported as a package.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predict_svm import predict_svm  # noqa: E402

# Not ready yet - uncomment once these are implemented and trained:
# from predict_naive_bayes import predict_naive_bayes
# from predict_logistic_regression import predict_logistic_regression


# --- model registry --------------------------------------------------------
# Single source of truth for which models the UI can offer. Flip "available"
# to True and plug in "predict_fn" once a model is trained and its predict_*
# function is ready - no other UI code needs to change.
MODEL_REGISTRY = {
    "SVM": {
        "available": True,
        "predict_fn": predict_svm,
        "status_note": "Trained and ready.",
    },
    "Naive Bayes": {
        "available": False,
        "predict_fn": None,
        "status_note": "Coming soon - model not trained yet.",
    },
    "Logistic Regression": {
        "available": False,
        "predict_fn": None,
        "status_note": "Coming soon - model not trained yet.",
    },
    "All Models": {
        "available": False,
        "predict_fn": None,
        "status_note": "Coming soon - requires every individual model to be ready.",
    },
}

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


# --- sidebar: project info, model status, pipeline summary --------------
with st.sidebar:
    st.header("ℹ️ Project Info")
    st.markdown(
        "- **Task:** Clickbait headline detection\n"
        "- **Language scope:** English headlines only\n"
        "- **Approach:** Classic NLP (TF-IDF + linear classifiers)"
    )

    st.header("📊 Model Status")
    for name, info in MODEL_REGISTRY.items():
        icon = "✅" if info["available"] else "🚧"
        st.markdown(f"{icon} **{name}** — {info['status_note']}")

    st.header("🔧 Pipeline Summary")
    st.markdown(
        "1. Raw headline input\n"
        "2. Text cleaning (lowercase, strip special characters)\n"
        "3. TF-IDF vectorization (`tfidf_vectorizer.pkl`)\n"
        "4. Trained model prediction (e.g. `svm.pkl`)\n"
        "5. Score, severity & explanation output"
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


def _format_model_label(name: str) -> str:
    return name if MODEL_REGISTRY[name]["available"] else f"{name}  (Coming Soon)"


selected_model = st.radio(
    "Model",
    options=model_names,
    format_func=_format_model_label,
    horizontal=True,
    label_visibility="collapsed",
)

if not MODEL_REGISTRY[selected_model]["available"]:
    st.caption(f"🚧 {selected_model} is not available yet - {MODEL_REGISTRY[selected_model]['status_note']}")


# --- analyze button --------------------------------------------------------
st.subheader("3. Analyze")
analyze_clicked = st.button("🔍 Analyze Headline", type="primary")


def display_result(result: dict, model_name: str) -> None:
    """Render the prediction result as cards. `result` follows the shape
    produced by utils.format_result(): model_name, prediction, clickbait_score,
    severity, influential_words.
    """
    prediction = result["prediction"]
    score = result["clickbait_score"]
    severity = result["severity"]
    words = result["influential_words"]

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
        st.markdown(
            " ".join(f":blue-badge[{w}]" for w in words)
        )
    else:
        st.caption("No influential words were found in this headline.")

    st.caption(f"Model used: **{result['model_name']}**")
    st.info(
        "ℹ️ The score, severity, and explanation above are generated directly "
        "from the selected trained model's learned weights on this headline - "
        "they are not based on a manually written trigger-word list."
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
    elif not MODEL_REGISTRY[selected_model]["available"]:
        st.info(f"🚧 {selected_model} is not available yet. Please select **SVM** for now.")
    else:
        predict_fn = MODEL_REGISTRY[selected_model]["predict_fn"]
        with st.spinner(f"Analyzing headline with {selected_model}..."):
            result = predict_fn(headline)
        display_result(result, selected_model)


# --- model comparison placeholder ---------------------------------------
st.divider()
st.subheader("📈 Model Comparison (Coming Soon)")
st.caption("Once all models are trained, this table will compare their performance.")

comparison_rows = [
    {
        "Model": name,
        "Status": "Available" if info["available"] else "Pending",
        "Accuracy": "—",
        "F1-score": "—",
    }
    for name, info in MODEL_REGISTRY.items()
    if name != "All Models"
]
st.table(comparison_rows)
