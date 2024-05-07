import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score


def data_preparation(obesity: pd.DataFrame) -> pd.DataFrame:
    obesity.rename(columns={"0be1dad": "OB_LEVEL"}, inplace=True)

    categorical_columns = ['Gender', 'MTRANS', 'OB_LEVEL']
    obesity[categorical_columns] = obesity[categorical_columns].astype('category')
    caec_mapping = {"0": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_mapping = {"0": 0, "Sometimes": 1, "Frequently": 2}
    ob_level_mapping = {"Insufficient_Weight": 0, "0rmal_Weight": 1, "Overweight_Level_I": 2, "Overweight_Level_II": 3,
                        "Obesity_Type_I": 4, "Obesity_Type_II": 5, "Obesity_Type_III": 6}
    obesity['CAEC'] = obesity['CAEC'].map(caec_mapping)
    obesity['CALC'] = obesity['CALC'].map(calc_mapping)
    obesity['OB_LEVEL'] = obesity['OB_LEVEL'].map(ob_level_mapping)
    obesity['CAEC'] = pd.Categorical(obesity['CAEC'], categories=[0, 1, 2, 3], ordered=True)
    obesity['CALC'] = pd.Categorical(obesity['CALC'], categories=[0, 1, 2], ordered=True)
    obesity['OB_LEVEL'] = pd.Categorical(obesity['OB_LEVEL'], categories=[0, 1, 2, 3, 4, 5, 6], ordered=True)

    numeric_columns = ['Age', 'Height', 'Weight', 'FCVC', 'CH2O', 'NCP', 'FAF', 'TUE']
    z_scores = stats.zscore(obesity[numeric_columns])
    threshold = 3
    outliers = (abs(z_scores) > threshold)
    outliers.any()

    return obesity


class ModelBean:
    def __init__(self, rfc, X_train, X_test, y_train, y_test):
        self.rfc = rfc
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


def create(obesity: pd.DataFrame) -> ModelBean:
    X, y = obesity.drop('OB_LEVEL', axis=1), obesity['OB_LEVEL']
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rfc = RandomForestClassifier(random_state=42, max_depth=30, min_samples_leaf=5, min_samples_split=2,
                                 n_estimators=100, max_features='log2')

    score = cross_val_score(rfc, X_train, y_train, cv=10)
    np.mean(score)
    return ModelBean(rfc, X_train, X_test, y_train, y_test)


def evaluate_model(model_bean: ModelBean) -> ModelBean:
    model_bean.rfc.fit(model_bean.X_train, model_bean.y_train)
    y_pred = model_bean.rfc.predict(model_bean.X_test)
    conf_matrix = confusion_matrix(model_bean.y_test, y_pred)

    print(conf_matrix)
    print(f"Precision: {accuracy_score(model_bean.y_test, y_pred)}")
    print(f"Recall: {recall_score(model_bean.y_test, y_pred, average='macro')}")
    print(f"F1 score: {f1_score(model_bean.y_test, y_pred, average='macro')}")
    return model_bean


def save_model(model: RandomForestClassifier, dir_path: str) -> None:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        dump(model, os.path.join(dir_path, 'obesity_classifier.joblib'))
        print('Model saved successfully')
    except Exception as e:
        print(f'Error while saving model: {e}')


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        'id': pd.api.types.is_integer_dtype,
        'Gender': pd.api.types.is_string_dtype,  # category
        'Age': pd.api.types.is_float_dtype,
        'Height': pd.api.types.is_float_dtype,
        'Weight': pd.api.types.is_float_dtype,
        'family_history_with_overweight': pd.api.types.is_integer_dtype,
        'FAVC': pd.api.types.is_integer_dtype,
        'FCVC': pd.api.types.is_float_dtype,
        'NCP': pd.api.types.is_float_dtype,
        'SMOKE': pd.api.types.is_integer_dtype,
        'CH2O': pd.api.types.is_float_dtype,
        'SCC': pd.api.types.is_integer_dtype,
        'FAF': pd.api.types.is_float_dtype,
        'TUE': pd.api.types.is_float_dtype,
        'CAEC': pd.api.types.is_categorical_dtype,  # category
        'CALC': pd.api.types.is_categorical_dtype,  # category
        'MTRANS': pd.api.types.is_categorical_dtype,  # category
        'OB_LEVEL': pd.api.types.is_categorical_dtype  # category
    }

    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for column_name, data_format in required_columns.items():
        assert data_format(data[column_name]), \
            f'Column {column_name} failed test {data_format}'

    ranges = {
        'CAEC': (0, 3),
        'CALC': (0, 2),
        'OB_LEVEL': (0, 6)
    }

    for column_name, (minimum, maximum) in ranges.items():
        assert data[column_name].dropna().between(minimum, maximum).all(), (
            f'Column {column_name} failed the test. Should be between {minimum} and {maximum},'
            f'instead min={data[column_name].min()} and max={data[column_name].max()}'
        )

    return data
