import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error)
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

def linear_regression(target: str,
                      core_features: list,
                      data_test: pd.DataFrame,
                      data_train: pd.DataFrame,
                      do_residuals: bool = True) -> None:
    """Linear regression model on weather data from NOAA."""

    # fit model with data
    regr = LinearRegression()
    regr.fit(data_train[core_features], data_train[target])
    predictions = regr.predict(data_test[core_features])
    residuals = np.array(data_test[target]) - predictions

    # get predictions and generate residuals and plot
    if do_residuals:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].scatter(predictions, residuals)
        axs[0].set_title('Residual Plot')

        axs[1].hist(residuals, bins=20, alpha=0.2)
        axs[1].set_title('Residual Histogram')

        plt.tight_layout()
        plt.show()

    # get r-squared from test data
    r_squared = regr.score(data_test[core_features], data_test[target])
    print(f'R-Squared (Linear Regression): {r_squared}')

    # get the mae
    mae = mean_absolute_error(data_test[target], predictions)
    print(f'Mean Absolute Error (Linear Regression): {mae}')

    # get the mse
    mse = mean_squared_error(data_test[target], predictions)
    print(f'Mean Squared Error (Linear Regression): {mse}')