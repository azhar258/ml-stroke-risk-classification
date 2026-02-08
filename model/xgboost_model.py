try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def train_xgboost(X_train, y_train):
    if XGBClassifier is None:
        return None

    model = XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model