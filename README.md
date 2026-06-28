# Clickbait and Sensationalism Detection with Severity Rating

## 1. Project Overview

This project develops a Natural Language Processing (NLP) system to detect clickbait and sensationalism in English online headlines. The system uses a labelled English clickbait dataset from Kaggle to train multiple supervised machine learning models and classify headlines as either Clickbait or Non-clickbait.

In addition to classification, the system provides:

* Model-based clickbait score (0–100)
* Severity rating: Low / Medium / High
* Model-based explanation using influential TF-IDF words
* Model selection: Naive Bayes, Logistic Regression, SVM, or All Models
* Majority voting when All Models is selected
* Model comparison table

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
User selects model:
  - Naive Bayes / Logistic Regression / SVM / All Models
↓
User inputs English headline
↓
Selected model predicts Clickbait or Non-clickbait
↓
Model-based scoring (clickbait probability × 100)
↓
Model-based severity (based on score range)
↓
Model-based explanation (influential TF-IDF words)
↓
Streamlit UI output
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
│   ├── shared_pipeline.py            # Dataset loading, cleaning, TF-IDF, train/test split
│   ├── preprocessing.py              # Text cleaning functions
│   ├── train_naive_bayes.py          # Train Naive Bayes model
│   ├── train_logistic_regression.py  # Train Logistic Regression model
│   ├── train_svm.py                  # Train SVM model
│   ├── predict_naive_bayes.py        # Predict using Naive Bayes
│   ├── predict_logistic_regression.py# Predict using Logistic Regression
│   ├── predict_svm.py                # Predict using SVM
│   ├── evaluate_models.py            # Shared evaluation functions
│   └── utils.py                      # Shared utility functions
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
├── reports/                          # Reports, charts, screenshots
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
| `shared_pipeline.py` | Shared data pipeline: load dataset, clean text, TF-IDF vectorization, train/test split. All models use the same pipeline. |
| `preprocessing.py` | Text cleaning functions: lowercase, remove punctuation, remove special characters, remove extra spaces. |
| `train_naive_bayes.py` | Train Naive Bayes model using shared TF-IDF features. Save as `naive_bayes.pkl`. |
| `train_logistic_regression.py` | Train Logistic Regression model using shared TF-IDF features. Save as `logistic_regression.pkl`. |
| `train_svm.py` | Train SVM (LinearSVC) model using shared TF-IDF features. Save as `svm.pkl`. |
| `predict_naive_bayes.py` | Predict function for Naive Bayes. Returns standard output format. |
| `predict_logistic_regression.py` | Predict function for Logistic Regression. Returns standard output format. |
| `predict_svm.py` | Predict function for SVM. Returns standard output format. |
| `evaluate_models.py` | Shared evaluation function: Accuracy, Precision, Recall, F1-score, confusion matrix. Model comparison table. |
| `utils.py` | Shared utility functions used across modules. |

### `app/`

| File | Description |
|------|-------------|
| `streamlit_app.py` | Streamlit web app: headline input, model selector, prediction display, score, severity, explanation, model comparison. |

### `models/`

| File | Description |
|------|-------------|
| `naive_bayes.pkl` | Saved Naive Bayes model |
| `logistic_regression.pkl` | Saved Logistic Regression model |
| `svm.pkl` | Saved SVM model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |

---

## 6. Standard Output Format

Each model's predict function must return the same output format:

```json
{
    "model_name": "Logistic Regression",
    "prediction": "Clickbait",
    "clickbait_score": 84,
    "severity": "High",
    "influential_words": ["shocking", "secret", "believe"]
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
0–30   = Low
31–65  = Medium
66–100 = High
```

### Explanation

Influential words are extracted from the selected model's coefficients / feature weights combined with the TF-IDF vectorizer features.

---

## 7. Dataset

The project uses a labelled English clickbait dataset from Kaggle.

Dataset columns:

```
headline, clickbait
```

Label meaning:

```
1 = Clickbait
0 = Non-clickbait
```

---

## 8. NLP and Machine Learning Methods

### Text Preprocessing

* Lowercase
* Remove punctuation
* Remove special characters
* Remove missing values
* Remove duplicate records

### Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) with n-gram support `(1, 3)` to capture both single words and phrases.

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

---

## 10. System Features

### Must Have

* English Kaggle clickbait dataset with headline and label columns
* Text preprocessing (lowercase, remove punctuation, remove special characters, remove missing/duplicate records)
* TF-IDF feature extraction
* Train 3 models: Naive Bayes, Logistic Regression, SVM
* Evaluate 3 models: Accuracy, Precision, Recall, F1-score
* Model comparison table
* Model-based clickbait score
* Model-based severity level: Low / Medium / High
* Model-based explanation: influential words from selected model
* Streamlit UI for single-title prediction

### Should Have

* Majority voting when All Models is selected
* Final score for All Models using average of 3 model scores
* Final severity based on average score
* Display individual results for each model when All Models is selected
* Confusion matrix for each model
* Model comparison chart
* Consistent UI layout for all model outputs

### Could Have

* Batch CSV prediction (upload CSV, predict all titles)
* Download prediction result as CSV
* PCA visualization (reduce TF-IDF features to 2D, show clickbait distribution)

---

## 11. Team Task Distribution

| Member | Module in Charge |
|--------|-----------------|
| Member 1 | Naive Bayes Classification: train, predict, score, severity, influential words |
| Member 2 | Logistic Regression Classification and Shared Model Evaluation |
| Member 3 | SVM Classification, Shared Pipeline, Majority Voting, Streamlit UI Integration |

---

## 12. How to Run

### Step 1: Install Dependencies

```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn joblib
```

### Step 2: Place Dataset

Place the Kaggle dataset in:

```
data/raw/clickbait_data.csv
```

### Step 3: Run Shared Pipeline

```bash
python src/shared_pipeline.py
```

### Step 4: Train Models

```bash
python src/train_naive_bayes.py
python src/train_logistic_regression.py
python src/train_svm.py
```

### Step 5: Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 13. Expected Output

For headline input:

```
You won't believe what happened next!
```

Single model output:

```
Prediction: Clickbait
Clickbait Score: 84/100
Severity: High
Influential Words: shocking, secret, believe
```

All Models output:

```
Naive Bayes:          Clickbait (Score: 82)
Logistic Regression:  Clickbait (Score: 84)
SVM:                  Non-clickbait (Score: 45)

Final Prediction: Clickbait (Majority Voting: 2/3)
Final Score: 70 (Average)
Final Severity: High
```

---

## 14. Notes

* This project focuses on English clickbait detection only.
* Scoring, severity, and explanation are model-based (not rule-based).
* SVM uses LinearSVC for coefficient-based explanation support.
* TF-IDF uses n-gram range (1, 3) to capture both words and phrases.
