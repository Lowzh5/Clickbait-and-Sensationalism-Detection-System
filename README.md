# Clickbait and Sensationalism Detection with Severity Rating

## 1. Project Overview

This project develops a Natural Language Processing (NLP) system to detect clickbait and sensationalism in English online headlines. The system uses a labelled English clickbait dataset from Kaggle to train multiple supervised machine learning models and classify headlines as either Clickbait or Non-clickbait.

In addition to classification, the system provides:

* Model-based clickbait score (0–100)
* Severity rating: Low / Medium / High
* Model-based explanation using influential TF-IDF words
* Model selection: Naive Bayes, Logistic Regression, SVM, or All Models
* Side-by-side comparison of all three models' predictions when All Models is selected

The prototype is built using Python and Streamlit.

---

## 2. Project Objectives

1. To preprocess online headline text for NLP analysis.
2. To convert text into numerical features using TF-IDF.
3. To train three text classification models for clickbait detection.
4. To compare model performance using Accuracy, Precision, Recall, and F1-score.
5. To generate a model-based clickbait score and classify severity.
6. To provide model-based explanation using influential words from TF-IDF features.
7. To develop a working Streamlit prototype for headline analysis.

---

## 3. System Flow

```
Kaggle labelled English clickbait dataset
↓
Text preprocessing
↓
TF-IDF feature extraction (with n-gram support)
↓
Train 3 models:
  - Naive Bayes
  - Logistic Regression
  - SVM
↓
Evaluate models:
  - Accuracy / Precision / Recall / F1-score
↓
User inputs English headline and selects model:
  - Naive Bayes / Logistic Regression / SVM / All Models
↓
Selected model(s) predict Clickbait or Non-clickbait
↓
Model-based scoring (clickbait probability × 100)
↓
Model-based severity (based on score range)
↓
Model-based explanation (influential TF-IDF words)
↓
Streamlit UI output (side by side per model if All Models is selected)
```
---

## 4. Project Structure

```text
Clickbait-detection-system/
│
├── data/
│   ├── raw/                          # Original Kaggle dataset
│   └── processed/                    # Cleaned dataset
│
├── src/
│   ├── data_pipeline.py              # Dataset loading, cleaning, TF-IDF, train/test split
│   ├── train_naive_bayes.py          # Train Naive Bayes model
│   ├── train_logistic_regression.py  # Train Logistic Regression model
│   ├── train_svm.py                  # Train SVM model
│   ├── predict_naive_bayes.py        # Predict using Naive Bayes
│   ├── predict_logistic_regression.py# Predict using Logistic Regression
│   ├── predict_svm.py                # Predict using SVM
│   ├── evaluation.py                 # Shared evaluation function (accuracy/precision/recall/F1/confusion matrix)
│   └── inference_utils.py            # Shared inference-time helpers (severity, influential words, result formatting)
│
├── app/
│   └── streamlit_app.py              # Streamlit web application
│
├── models/                           # Saved .pkl model files
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   └── tfidf_vectorizer.pkl
│
├── exploration/                      # Ablation studies (not part of the app's runtime pipeline)
│   ├── tfidf_ablation_study.py       # Tests 18 TF-IDF configs x 3 models
│   ├── best_ablation_configuration.py# Averages F1 across models per config
│   ├── svm_c_ablation_study.py       # Tests SVM C values
│   ├── ablation_results.csv
│   ├── best_ablation_configuration_results.csv
│   └── svm_c_ablation_results.csv
│
├── reports/                          # Report charts and the scripts that generate them
│   ├── Image/                        # Generated PNG charts used in the report
│   └── visualization/                # Matplotlib scripts that generate the charts in Image/
│
├── README_tutor_version/             # Minimal install/run guide for the tutor
│   └── README.md
│
├── requirements.txt                  # Pinned dependency versions
└── README.md
```

---

## 5. File Descriptions

### `data/`

| Path | Description |
|------|-------------|
| `data/raw/clickbait_data.csv` | Original Kaggle clickbait dataset |
| `data/processed/cleaned_clickbait_data.csv` | Cleaned and preprocessed dataset |

### `src/`

| File | Description |
|------|-------------|
| `data_pipeline.py` | Shared data pipeline: load dataset, clean text, TF-IDF vectorization, train/test split. All models use the same pipeline. |
| `train_naive_bayes.py` | Train Naive Bayes model using shared TF-IDF features. Save as `naive_bayes.pkl`. |
| `train_logistic_regression.py` | Train Logistic Regression model using shared TF-IDF features. Save as `logistic_regression.pkl`. |
| `train_svm.py` | Train SVM (LinearSVC) model using shared TF-IDF features. Save as `svm.pkl`. |
| `predict_naive_bayes.py` | Predict function for Naive Bayes. Returns standard output format. |
| `predict_logistic_regression.py` | Predict function for Logistic Regression. Returns standard output format. |
| `predict_svm.py` | Predict function for SVM. Returns standard output format. |
| `evaluation.py` | Shared evaluation function: Accuracy, Precision, Recall, F1-score, confusion matrix. Used by all `train_*.py` scripts. |
| `inference_utils.py` | Shared inference-time helpers used by all `predict_*.py` scripts: severity rating, influential words, result formatting. |

### `app/`

| File | Description |
|------|-------------|
| `streamlit_app.py` | Streamlit web app: headline input, model selector, prediction display, score, severity, explanation, side-by-side All Models comparison. |

### `models/`

| File | Description |
|------|-------------|
| `naive_bayes.pkl` | Saved Naive Bayes model |
| `logistic_regression.pkl` | Saved Logistic Regression model |
| `svm.pkl` | Saved SVM model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |

### `exploration/`

Supports the ablation study in the report (Section 4.1). Fits everything in-memory and never touches `models/*.pkl`, so it's safe to re-run without affecting the app.

| File | Description |
|------|-------------|
| `tfidf_ablation_study.py` | Trains all 3 models under 18 TF-IDF configs (n-gram range x stop words x max_features); saves `ablation_results.csv`. |
| `best_ablation_configuration.py` | Averages each config's F1-score across the 3 models to find the overall best shared config; saves `best_ablation_configuration_results.csv`. |
| `svm_c_ablation_study.py` | Tests SVM at C = 0.01/0.1/1/10/100 with the TF-IDF config fixed; saves `svm_c_ablation_results.csv`. |

### `reports/`

| Path | Description |
|------|-------------|
| `reports/Image/` | Generated PNG charts (confusion matrices, ROC curves, influential-words bar charts, word-usage charts) used directly in the report. |
| `reports/visualization/` | Matplotlib scripts that generate each chart in `reports/Image/`. |

---

## 6. Standard Output Format

Each model's predict function must return the same output format:

```json
{
    "model_name": "Logistic Regression",
    "prediction": "Clickbait",
    "clickbait_score": 84,
    "severity": "High",
    "influential_words": ["shocking", "secret", "believe"],
    "influential_word_scores": [0.42, 0.31, 0.27]
}
```

### Scoring

Clickbait score is calculated from the model's clickbait probability / confidence:

```
Clickbait Score = Clickbait probability × 100
```

### Severity

Severity is assigned based on the clickbait score range:

```
0–33   = Low
34–66  = Medium
67–100 = High
```

### Explanation

Influential words are extracted from the selected model's coefficients / feature weights combined with the TF-IDF vectorizer features.

---

## 7. Dataset

The project uses a labelled English clickbait dataset from Kaggle.

Dataset columns: headline, clickbait

Clickbait label meaning:
1 = Clickbait
0 = Non-clickbait

---

## 8. NLP and Machine Learning Methods

### Text Preprocessing

* Lowercase
* Remove punctuation
* Remove special characters
* Remove missing values
* Remove duplicate records

### Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) with n-gram range `(1, 2)` and `max_features = 50,000` to capture both single words and two-word phrases.

### Models

| Model | Description |
|-------|-------------|
| Naive Bayes | Classifies based on word occurrence probability |
| Logistic Regression | Assigns weights to each word, calculates clickbait probability |
| SVM (LinearSVC) | Finds a decision boundary to separate clickbait from non-clickbait |

All models use the same shared TF-IDF features for fair comparison.

---

## 9. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Of predicted clickbait, how many are actually clickbait |
| Recall | Of actual clickbait, how many were detected |
| F1-score | Harmonic mean of Precision and Recall |