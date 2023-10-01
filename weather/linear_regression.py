import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from weather import utils

def linear_regression(target: str,
                      core_features: list,
                      data_test: pd.DataFrame,
                      data_train: pd.DataFrame,
                      do_residuals: bool = False) -> None:
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

    # generate all evals
    utils.generate_evals(target=target,
                         predictions=predictions,
                         data_test=data_test)