"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.3
"""

import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score


class ModelBean:
    def __init__(self, model, performance, X_train, X_test, y_train, y_test):
        self.model = model
        self.performance: float = performance
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


def load_champion(model_bean):
    if os.path.exists("src\\lab_4\\Model_champion\\champion.joblib"):
        print(100 * "-")
        print("Champion exists")
        print(100 * "-")
        return load("src\\lab_4\\Model_champion\\champion.joblib")
    else:
        return ModelBean(model=None, X_train=pd.DataFrame(), X_test=pd.DataFrame(), y_train=pd.DataFrame(),
                         y_test=pd.DataFrame(), performance=-1.0)


def compare_models(champion: ModelBean, challenger: ModelBean):
    challenger_perf = challenger.performance
    if champion.performance < 0:
        print(100 * "-")
        print('Champion does not exist, challenger is new champion')
        print(f'Challenger accuracy {challenger_perf}')
        print(100 * "-")
        dump(
            ModelBean(
                challenger.model,
                challenger.performance,
                challenger.X_train,
                challenger.X_test,
                challenger.y_train,
                challenger.y_test
            ),
            'src\\lab_4\\Model_champion\\champion.joblib'
        )
        dump(challenger.model, 'src\\lab_4\\Model_champion\\prod_champion.joblib')
        return

    champion_perf = champion.performance
    print(100 * "-")
    print(100 * "-")
    print(f'Champion accuracy {champion_perf}')
    print(f'Challenger accuracy {challenger_perf}')
    print(100 * "-")
    print(100 * "-")

    if challenger_perf > champion_perf:
        print('Challenger was better than champion. Replacing algorithms')
        dump(
            ModelBean(
                challenger.model,
                challenger.performance,
                challenger.X_train,
                challenger.X_test,
                challenger.y_train,
                challenger.y_test
            ),
            'src\\lab_4\\Model_champion\\champion.joblib'
        )
        dump(challenger.model, 'src\\lab_4\\Model_champion\\prod_champion.joblib')
        return
    print('No changes. Champion is better than challenger')


def evaluate_model(model_bean: ModelBean) -> ModelBean:
    model_bean.model.fit(
        np.array(model_bean.X_train.values), np.array(model_bean.y_train.values)
    )
    y_pred = model_bean.model.predict(model_bean.X_test.values)
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
        dump(model, os.path.join(dir_path, "obesity_classifier.joblib"))
        print("Model saved successfully")
    except Exception as e:
        print(f"Error while saving model: {e}")
