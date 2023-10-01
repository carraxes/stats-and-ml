import pandas as pd
from typing import Tuple
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error)

def generate_test_train(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train and test set from time series data."""
    data_train = data[data.index < pd.to_datetime('6/1/2017')]
    data_test = data[data.index >= pd.to_datetime('6/1/2017')]
    return data_train, data_test

def generate_evals(target: str, predictions: list, data_test: pd.DataFrame) -> None:
    """Generate evals after fitting and predicting values."""

    # get r-squared from test data
    r_squared = r2_score(data_test[target], predictions)
    print(f'R-Squared (Linear Regression): {r_squared}')

    # get the mae
    mae = mean_absolute_error(data_test[target], predictions)
    print(f'Mean Absolute Error (Linear Regression): {mae}')

    # get the mse
    mse = mean_squared_error(data_test[target], predictions)
    print(f'Mean Squared Error (Linear Regression): {mse}')