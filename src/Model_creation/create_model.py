"""
Creation of a model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier


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


if __name__ == '__main__':
    pass
