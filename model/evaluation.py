from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    # Predict the target values for the test set
    y_pred = model.predict(X_test)

    # Check if the model supports probability prediction for AUC calculation
    if hasattr(model, "predict_proba"):
        # Get the predicted probabilities for the positive class
        y_prob = model.predict_proba(X_test)[:, 1]
        # Calculate AUC score
        auc = roc_auc_score(y_test, y_prob)
    else:
        # If not supported, set AUC to None
        auc = None

    # Return a dictionary of evaluation metrics
    return {
        "Accuracy": accuracy_score(y_test, y_pred),  
        "AUC": auc,  
        "Precision": precision_score(y_test, y_pred), 
        "Recall": recall_score(y_test, y_pred),  
        "F1": f1_score(y_test, y_pred),  
        "MCC": matthews_corrcoef(y_test, y_pred),  
        "ConfusionMatrix": confusion_matrix(y_test, y_pred)  
    }