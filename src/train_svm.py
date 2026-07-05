import os
import joblib
from sklearn.svm import LinearSVC
from shared_pipeline import BASE_DIR, load_dataset, clean_dataset, tfidf
from utils import evaluate_model

if __name__ == "__main__":
    # reuse the shared pipeline instead of duplicating loading/cleaning/TF-IDF code
    df = load_dataset()
    cleaned_df = clean_dataset(df)
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = tfidf(cleaned_df)

    # LinearSVC = soft-margin Linear SVM.
    model = LinearSVC(C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # evaluate on the existing test split (must .fit() model first)
    y_pred = model.predict(X_test_tfidf)
    evaluate_model(y_test, y_pred, model_name="SVM (LinearSVC)")

    # save trained model
    model_path = os.path.join(BASE_DIR, "models", "svm.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
