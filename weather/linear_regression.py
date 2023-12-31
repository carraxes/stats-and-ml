import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from weather.utils import (
    generate_evals,
    generate_residual_plot
)

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
        generate_residual_plot(residuals=residuals,
                               predictions=predictions)

    # generate all evals
    generate_evals(target=target,
                   predictions=predictions,
                   data_test=data_test)