import os
import joblib
from sklearn.linear_model import LogisticRegression
from data_pipeline import BASE_DIR, load_dataset, clean_dataset, tfidf
from evaluation import evaluate_model

if __name__ == "__main__":
    # reuse the shared pipeline instead of duplicating loading/cleaning/TF-IDF code
    df = load_dataset()
    cleaned_df = clean_dataset(df)
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = tfidf(cleaned_df)

    # LogisticRegression = linear model that outputs calibrated probabilities directly
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # evaluate on the existing test split (must .fit() model first)
    y_pred = model.predict(X_test_tfidf)
    evaluate_model(y_test, y_pred, model_name="Logistic Regression")

    # save trained model
    model_path = os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")