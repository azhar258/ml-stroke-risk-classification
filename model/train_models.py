import pandas as pd

from model.preprocessing import load_and_preprocess_data
from model.evaluation import evaluate_model

from model.logistic_model import train_logistic_regression
from model.decision_tree_model import train_decision_tree
from model.naive_bayes_model import train_naive_bayes
from model.random_forest_model import train_random_forest
from model.knn_model import train_knn
from model.xgboost_model import train_xgboost


def train_all_models(csv_path):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_path)

    models = {
        'Logistic Regression': train_logistic_regression(X_train, y_train),
        'Decision Tree': train_decision_tree(X_train, y_train),
        'Naive Bayes': train_naive_bayes(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'KNN': train_knn(X_train, y_train),
    }

    # XGBoost handled separately (important!)
    xgb_model = train_xgboost(X_train, y_train)
    if xgb_model is not None:
        models['XGBoost'] = xgb_model

    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        results.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "AUC": metrics["AUC"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "MCC": metrics["MCC"]
        })

    return pd.DataFrame(results), models