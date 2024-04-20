"""
Evaluation of a model
"""

import wandb
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from Model_creation.create_model import ModelBean


def evaluate_model(
    model_bean: ModelBean,
    api_key: str,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    n_estimators,
) -> None:
    model_bean.rfc.fit(model_bean.X_train, model_bean.y_train)
    y_pred = model_bean.rfc.predict(model_bean.X_test)
    y_probas = model_bean.rfc.predict_proba(model_bean.X_test)
    labels = [0, 1, 2, 3, 4, 5, 6]
    conf_matrix = confusion_matrix(model_bean.y_test, y_pred)

    print(conf_matrix)
    print(f"Precision: {accuracy_score(model_bean.y_test, y_pred)}")
    print(f"Recall: {recall_score(model_bean.y_test, y_pred, average='macro')}")
    print(f"F1 score: {f1_score(model_bean.y_test, y_pred, average='macro')}")
    wandb.login(key=api_key)
    wandb.init(
        project="PJATK-ASI-14C",
        config={
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "n_estimators": n_estimators,
        },
    )
    wandb.sklearn.plot_classifier(
        model_bean.rfc,
        model_bean.X_train,
        model_bean.X_test,
        model_bean.y_train,
        model_bean.y_test,
        y_pred,
        y_probas,
        labels,
        model_name="Random Forest Classifier",
    )
    wandb.sklearn.plot_roc(model_bean.y_test, y_probas, labels)
    wandb.sklearn.plot_precision_recall(model_bean.y_test, y_probas, labels)
    wandb.sklearn.plot_confusion_matrix(model_bean.y_test, y_pred, labels)
    wandb.sklearn.plot_summary_metrics(
        model_bean.rfc,
        model_bean.X_train,
        model_bean.y_train,
        model_bean.X_test,
        model_bean.y_test,
    )


if __name__ == "__main__":
    pass
