# Clickbait and Sensationalism Detection System

**Group Number:** 1

A Streamlit-based system that classifies news headlines as Clickbait or Non-clickbait using three machine learning models (Naive Bayes, Logistic Regression, SVM), with a clickbait score, severity level, and influential-word explanation.

Tested with Python 3.13.1 on Windows.

---

## 1. Libraries to Install

Run each command below one by one in the project root folder:

1. `pip install pandas`
2. `pip install numpy`
3. `pip install scikit-learn`
4. `pip install joblib`
5. `pip install matplotlib`
6. `pip install altair`
7. `pip install streamlit`

(Alternatively, run `pip install -r requirements.txt` to install all of them at once.)

---

## 2. How to Run

The trained models are already saved in `models/`, so training is optional. Run the files in this order:

1. `python src/train_naive_bayes.py` — trains and saves `models/naive_bayes.pkl`
2. `python src/train_logistic_regression.py` — trains and saves `models/logistic_regression.pkl`
3. `python src/train_svm.py` — trains and saves `models/svm.pkl`
4. `streamlit run app/streamlit_app.py` — launches the web app

Notes:
- Steps 1–3 also save `models/tfidf_vectorizer.pkl` and can be skipped if you just want to use the already-trained models.
- `src/data_pipeline.py` does not need to be run separately — it is automatically called by steps 1–3.
- After step 4, Streamlit prints a local URL (e.g. `http://localhost:8501`). Open it in a browser, enter a headline, select a model (or All Models), and view the prediction, score, severity, and influential words.
