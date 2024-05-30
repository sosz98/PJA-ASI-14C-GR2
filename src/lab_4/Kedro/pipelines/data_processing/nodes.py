import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from autogluon.tabular import TabularPredictor


class ModelBean:
    def __init__(self, challenger, performance, X_train, X_test, y_train, y_test):
        self.challenger = challenger
        self.performance = performance
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


def create(obesity: pd.DataFrame) -> ModelBean:
    X, y = obesity.drop("OB_LEVEL", axis=1), obesity["OB_LEVEL"]
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Combine X_train and y_train back into a single DataFrame for AutoGluon
    train_data = X_train.copy()
    train_data["OB_LEVEL"] = y_train

    # Combine X_test and y_test back into a single DataFrame for AutoGluon
    test_data = X_test.copy()
    test_data["OB_LEVEL"] = y_test

    challenger_model = TabularPredictor(label="OB_LEVEL").fit(
        train_data,
        hyperparameters={
            "GBM": {},
            "RF": {},
        },
        presets="medium_quality_faster_train",
    )

    # Load the best model
    performance = challenger_model.evaluate(train_data)
    print(100 * "-")
    print(performance)
    print(100 * "-")
    return ModelBean(challenger_model, performance, X_train, X_test, y_train, y_test)


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "id": pd.api.types.is_integer_dtype,
        "Gender": pd.api.types.is_string_dtype,  # category
        "Age": pd.api.types.is_float_dtype,
        "Height": pd.api.types.is_float_dtype,
        "Weight": pd.api.types.is_float_dtype,
        "family_history_with_overweight": pd.api.types.is_integer_dtype,
        "FAVC": pd.api.types.is_integer_dtype,
        "FCVC": pd.api.types.is_float_dtype,
        "NCP": pd.api.types.is_float_dtype,
        "SMOKE": pd.api.types.is_integer_dtype,
        "CH2O": pd.api.types.is_float_dtype,
        "SCC": pd.api.types.is_integer_dtype,
        "FAF": pd.api.types.is_float_dtype,
        "TUE": pd.api.types.is_float_dtype,
    }

    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for column_name, data_format in required_columns.items():
        assert data_format(
            data[column_name]
        ), f"Column {column_name} failed test {data_format}"

    ranges = {"CAEC": (0, 3), "CALC": (0, 2), "OB_LEVEL": (0, 6)}

    for column_name, (minimum, maximum) in ranges.items():
        assert data[column_name].dropna().between(minimum, maximum).all(), (
            f"Column {column_name} failed the test. Should be between {minimum} and {maximum},"
            f"instead min={data[column_name].min()} and max={data[column_name].max()}"
        )

    return data
