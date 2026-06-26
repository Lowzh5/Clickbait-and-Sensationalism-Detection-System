# Clickbait and Sensationalism Detection with Severity Rating Using NLP

## 1. Project Overview

This project develops a Natural Language Processing (NLP) system to detect clickbait and sensationalism in online headlines. The system uses a labelled English clickbait dataset to train multiple supervised machine learning models and classify headlines as either clickbait or non-clickbait.

In addition to basic clickbait detection, the system also provides:

* Clickbait score from 0 to 100
* Severity rating: Low, Medium, or High
* Explanation of detected clickbait triggers
* Model selection between Naive Bayes, Logistic Regression, SVM, or All Models
* Majority voting when all models are selected

The prototype is built using Python and Streamlit.

---

## 2. Project Objectives

The objectives of this project are:

1. To preprocess online headline text for NLP analysis.
2. To convert text into numerical features using TF-IDF.
3. To train multiple text classification models for clickbait detection.
4. To compare model performance using Accuracy, Precision, Recall, and F1-score.
5. To calculate a clickbait score and classify sensationalism severity.
6. To provide explainable reasons for clickbait classification.
7. To develop a working prototype for headline analysis.

---

## 3. Project Structure

```text
clickbait-nlp-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── shared_pipeline.py
│   ├── train_naive_bayes.py
│   ├── train_logistic_regression.py
│   ├── train_svm.py
│   ├── evaluate_models.py
│   ├── scoring.py
│   ├── explanation.py
│   └── predict.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   └── tfidf_vectorizer.pkl
│
├── reports/
└── README.md
```

---

## 4. Folder and File Description

### `data/`

This folder stores the dataset used in the project.

#### `data/raw/`

Stores the original dataset downloaded from Kaggle.

Example:

```text
clickbait_data.csv
```

The raw dataset should not be modified directly.

#### `data/processed/`

Stores the cleaned and preprocessed version of the dataset.

Example:

```text
cleaned_clickbait_data.csv
```

---

### `src/`

This folder stores the main Python source code.

#### `shared_pipeline.py`

Contains the shared data processing pipeline used by all models.

Main functions:

* Load dataset
* Clean headline text
* Split data into training and testing sets
* Apply TF-IDF feature extraction
* Return `X_train`, `X_test`, `y_train`, and `y_test`

This ensures all models use the same preprocessing and feature extraction process.

#### `train_naive_bayes.py`

Trains the Naive Bayes model using TF-IDF features.

Main tasks:

* Import shared pipeline
* Train Naive Bayes model
* Predict test data
* Evaluate performance
* Save trained model as `naive_bayes.pkl`

#### `train_logistic_regression.py`

Trains the Logistic Regression model using TF-IDF features.

Main tasks:

* Import shared pipeline
* Train Logistic Regression model
* Predict test data
* Evaluate performance
* Save trained model as `logistic_regression.pkl`

#### `train_svm.py`

Trains the Support Vector Machine (SVM) model using TF-IDF features.

Main tasks:

* Import shared pipeline
* Train SVM model
* Predict test data
* Evaluate performance
* Save trained model as `svm.pkl`

#### `evaluate_models.py`

Contains shared evaluation functions for all models.

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Each model imports this file to evaluate its performance using the same evaluation logic.

#### `scoring.py`

Contains the clickbait scoring and severity rating logic.

Main tasks:

* Detect keyword triggers
* Calculate clickbait score from 0 to 100
* Assign severity level:

  * Low
  * Medium
  * High

Example severity mapping:

```text
0–30   = Low
31–65  = Medium
66–100 = High
```

#### `explanation.py`

Generates explanation reasons for the prediction.

Example explanations:

* Detected clickbait phrase: "you won't believe"
* Detected sensational phrase: "what happened next"
* Detected exclamation mark
* The selected model classified the headline as clickbait

#### `predict.py`

Combines the trained model, TF-IDF vectorizer, scoring module, and explanation module.

Main tasks:

* Load trained model
* Load TF-IDF vectorizer
* Transform user input headline
* Predict clickbait or non-clickbait
* Calculate score
* Assign severity
* Generate explanation
* Return final prediction output

---

### `app/`

This folder contains the Streamlit prototype.

#### `streamlit_app.py`

The main web application file.

Main features:

* Headline input box
* Model selector:

  * Naive Bayes
  * Logistic Regression
  * SVM
  * All Models
* Prediction output
* Clickbait score
* Severity level
* Explanation reasons
* Model comparison table
* Majority voting result when All Models is selected

---

### `models/`

This folder stores trained machine learning models and the TF-IDF vectorizer.

#### `naive_bayes.pkl`

Saved Naive Bayes model.

#### `logistic_regression.pkl`

Saved Logistic Regression model.

#### `svm.pkl`

Saved SVM model.

#### `tfidf_vectorizer.pkl`

Saved TF-IDF vectorizer used to transform new headline input.

---

### `reports/`

This folder stores report-related files.

Examples:

* Final documentation
* Model comparison table
* Confusion matrix images
* Prototype screenshots
* Result charts
* AI disclosure notes

---

## 5. Dataset

The project uses a labelled English clickbait dataset from Kaggle.

Expected dataset columns:

```text
headline, clickbait
```

Label meaning:

```text
1 = Clickbait
0 = Non-clickbait
```

The dataset is used for supervised NLP text classification.

---

## 6. NLP and Machine Learning Methods

### Text Preprocessing

The project applies text preprocessing steps such as:

* Lowercasing
* Removing punctuation
* Removing special characters
* Removing missing values
* Removing duplicated records

### Feature Extraction

The project uses TF-IDF to convert headline text into numerical features.

TF-IDF is used because machine learning models cannot directly process raw text. It assigns numerical importance to words based on their frequency in a headline and their rarity across the dataset.

### Models

The project compares three machine learning models:

1. Naive Bayes
2. Logistic Regression
3. Support Vector Machine (SVM)

All models use the same TF-IDF features for fair comparison.

---

## 7. Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

These metrics are used to compare the performance of Naive Bayes, Logistic Regression, and SVM.

---

## 8. System Features

### Must-Have Features

* English clickbait dataset
* Text preprocessing
* TF-IDF feature extraction
* Naive Bayes model
* Logistic Regression model
* SVM model
* Model evaluation
* Streamlit UI
* Single headline prediction

### Should-Have Features

* Clickbait score
* Low / Medium / High severity rating
* Explanation reason
* Model selector
* Majority voting using all models
* Model comparison table

### Future Enhancements

* Chinese clickbait detection
* Chinese word segmentation
* Multilingual English-Chinese model
* Batch CSV prediction
* Neutral headline rewriting
* Web crawler for automatic headline collection
* Transformer-based model such as BERT

---

## 9. Team Task Distribution

| Member   | Module in Charge                                                                   |
| -------- | ---------------------------------------------------------------------------------- |
| Member 1 | Naive Bayes Classification, Clickbait Scoring, and Severity Rating                 |
| Member 2 | Logistic Regression Classification and Model Performance Evaluation                |
| Member 3 | SVM Classification, Explanation Module, Majority Voting, and Prototype Integration |

---

## 10. How to Run the Project

### Step 1: Install Dependencies

```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn joblib
```

### Step 2: Place Dataset

Place the Kaggle dataset inside:

```text
data/raw/
```

Example:

```text
data/raw/clickbait_data.csv
```

### Step 3: Train Models

Run the model training scripts:

```bash
python src/train_naive_bayes.py
python src/train_logistic_regression.py
python src/train_svm.py
```

### Step 4: Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 11. Expected Output

For a sample headline:

```text
You won't believe what happened next!
```

The system may output:

```text
Prediction: Clickbait
Clickbait Score: 88/100
Severity: High

Explanation:
- Detected phrase: "you won't believe"
- Detected phrase: "what happened next"
- Detected exclamation mark
```

If All Models is selected, the system may output:

```text
Naive Bayes: Clickbait
Logistic Regression: Clickbait
SVM: Non-clickbait

Final Prediction: Clickbait
Reason: 2 out of 3 models predicted Clickbait
```

---

## 12. Notes

This project focuses on English clickbait detection using supervised NLP text classification. Chinese clickbait detection and multilingual model training are considered future enhancements.
