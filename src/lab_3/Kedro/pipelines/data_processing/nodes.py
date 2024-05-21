import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score


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
