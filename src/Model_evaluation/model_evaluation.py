"""
Evaluation of a model
"""

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from src.Model_creation.create_model import ModelBean


def evaluate_model(model_bean: ModelBean) -> None:
    model_bean.rfc.fit(model_bean.X_train, model_bean.y_train)
    y_pred = model_bean.rfc.predict(model_bean.X_test)
    conf_matrix = confusion_matrix(model_bean.y_test, y_pred)

    print(conf_matrix)
    print(f"Precision: {accuracy_score(model_bean.y_test, y_pred)}")
    print(f"Recall: {recall_score(model_bean.y_test, y_pred, average='macro')}")
    print(f"F1 score: {f1_score(model_bean.y_test, y_pred, average='macro')}")


if __name__ == '__main__':
    pass
