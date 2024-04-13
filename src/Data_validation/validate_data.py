"""
Validate columns and types of dataset
"""
import pandas as pd


def validate_column_types(data: pd.DataFrame) -> None:

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


def validate_categorical_column_ranges(data: pd.DataFrame) -> None:

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
