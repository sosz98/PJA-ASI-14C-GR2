"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.3
"""

import pandas as pd
from scipy.stats import stats


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
