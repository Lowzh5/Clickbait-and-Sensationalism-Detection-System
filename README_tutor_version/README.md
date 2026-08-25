# Clickbait and Sensationalism Detection System

A Streamlit-based system that classifies news headlines as Clickbait or Non-clickbait using three machine learning models (Naive Bayes, Logistic Regression, SVM), with a clickbait score, severity level, and influential-word explanation.

Tested with Python 3.13.1 on Windows.

---

## 1. Libraries to Install

Run this command in the project root folder to install all required libraries at the exact versions this project was tested with:

`pip install -r requirements.txt`

This installs: pandas, numpy, scikit-learn, joblib, matplotlib, altair, streamlit.

---

## 2. How to Run

The trained models are already saved in `models/`, so training is optional. However, Run these files in this order if want to go through step by step(1 to 4):

1. `python src/train_naive_bayes.py` — trains and saves the model to modelss/naive_bayes.pkl`
2. `python src/train_logistic_regression.py` — trains and saves the model to models/logistic_regression.pkl
3. `python src/train_svm.py` — trains and saves the model to models/svm.pkl
4. `python -m streamlit run app/streamlit_app.py` — launches the web app 

OR directly go to step 4
 
Notes:
- `src/data_pipeline.py` does not need to be run separately — it is automatically called by steps 1–3.
- After step 4, Streamlit prints a local URL (e.g. `http://localhost:8501`). Open it in a browser, enter a headline (example provided in the web page), select a model (or All Models), and view the prediction, score, severity, and influential words.
- The `reports` and `exploration` folders are used for assignment-related reports and data visualizations. They are not part of the system process flow.

---

## 3. Known Limitations

- Only supports **English** headlines.
- Predictions are based on the headline text only, not the full article content.
- Very short headlines (under 10 characters) are rejected by input validation.
- The clickbait score/severity is a model confidence estimate, not a guaranteed fact.

---

## 4. Troubleshooting

Problem: Solution/Justification

- `ModuleNotFoundError`: Run `pip install -r requirements.txt`. 
- `FileNotFoundError` on `models/*.pkl`: Run steps 1–3 in Section 2 to train and save the models first. 
- Port `8501` already in use: Stop the previous process first. Find its corresponting PID with `netstat -ano | findstr :8501`, then run `taskkill /PID <pid> /F` (or press Ctrl+C in the terminal that's running it, if it's still open).
Note:
power shell terminal: `taskkill /PID <pid> /F`
git bash terminal:`taskkill //PID <pid> //F`
- "Please enter a headline" / "too short" warning: Not a bug — the app requires at least 10 characters of input.
