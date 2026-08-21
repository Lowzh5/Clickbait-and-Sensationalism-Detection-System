"""evaluate_model(y_test, y_pred, model_name) - prints and returns accuracy, precision, recall, f1, confusion matrix"""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

def evaluate_model(y_test, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{model_name} evaluation on test set:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:") # support = how many test set inside the class
    print(classification_report(y_test, y_pred, target_names=["Non-clickbait", "Clickbait"]))

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
