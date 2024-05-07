"""
Import dataset
"""

import pandas as pd

# from src import main


def collect_data(file_path: str) -> pd.DataFrame:
    try:
        obesity = pd.read_csv(file_path)
        return obesity
    except FileNotFoundError as err:
        print(f"Error while looking for file {err}")
        raise


if __name__ == "__main__":
    collect_data()
