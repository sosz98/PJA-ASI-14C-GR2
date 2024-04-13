"""
Import dataset
"""

import pandas as pd


def collect_data() -> pd.DataFrame:
    try:
        obesity = pd.read_csv('../obesity_level.csv')
        return obesity
    except FileNotFoundError as err:
        print(f'Error while looking for file {err}')
        raise


if __name__ == '__main__':
    collect_data()
