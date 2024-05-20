"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.3
"""

import os
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score


class ModelBean:
    def __init__(self, rfc, X_train, X_test, y_train, y_test):
        self.rfc = rfc
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


def evaluate_model(model_bean: ModelBean) -> ModelBean:
    model_bean.rfc.fit(np.array(model_bean.X_train.values), np.array(model_bean.y_train.values))
    y_pred = model_bean.rfc.predict(model_bean.X_test.values)
    conf_matrix = confusion_matrix(list(model_bean.y_test), y_pred)

    print(conf_matrix)
    print(f"Precision: {accuracy_score(list(model_bean.y_test), y_pred)}")
    print(f"Recall: {recall_score(list(model_bean.y_test), y_pred, average='macro')}")
    print(f"F1 score: {f1_score(list(model_bean.y_test), y_pred, average='macro')}")
    return model_bean


def save_model(model: RandomForestClassifier, dir_path: str) -> None:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        dump(model, os.path.join(dir_path, 'obesity_classifier.joblib'))
        print('Model saved successfully')
    except Exception as e:
        print(f'Error while saving model: {e}')