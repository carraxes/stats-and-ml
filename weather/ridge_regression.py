import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

def ridge_regression(target: str,
                     core_features: list,
                     data_test: pd.DataFrame,
                     data_train: pd.DataFrame,
                     do_residuals: bool = False) -> None:
    """"Ridge regression model on weather data from NOAA."""
    pass
    
    # # generate 50 alphas values using logspace
    # alphas_to_test = np.logspace(-4, 4, 50)
    
    # # ridge regression model with built-in cross-validation
    # ridge_cv = RidgeCV(alphas=alphas, cv=5)

    # # fit model
    # ridge_cv.fit(X, y)

    # # optimal regularization parameter
    # optimal_alpha = ridge_cv.alpha_
    # print(optimal_alpha)
