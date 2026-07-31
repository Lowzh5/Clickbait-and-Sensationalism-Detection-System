import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from data_pipeline import BASE_DIR, load_dataset, clean_dataset, tfidf
from evaluation import evaluate_model

if __name__ == "__main__":

    # Reuse shared pipeline to load and clean and vectorize the data
    df = load_dataset()
    cleaned_df = clean_dataset(df)
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = tfidf(cleaned_df)

    # Train Multinomial Naive Bayes model
    model = MultinomialNB(alpha = 0.1)
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    evaluate_model(y_test, y_pred, model_name = "Naive Bayes")

    # Save the trained model and vectorizer
    model_path = os.path.join(BASE_DIR, "models", "naive_bayes.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")