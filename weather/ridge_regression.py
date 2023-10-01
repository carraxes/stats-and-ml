import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from weather import utils

def ridge_regression(target: str,
                     core_features: list,
                     data: pd.DataFrame,
                     do_residuals: bool = False) -> None:
    """"Ridge regression model on weather data from NOAA."""
    alphas = np.logspace(-4, 4, 50)
    data_x = data[core_features].values
    data_y = data[[target]].values
    
    models = {}
    tscv = TimeSeriesSplit(n_splits=5)
    for alpha in alphas:
        errors = []
        for train_index, test_index in tscv.split(data_x):
            X_train, X_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
            
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            error = mean_squared_error(y_test, predictions)
            errors.append(error)
        models.update({min(errors): model})
    best_model = [models[i] for i in models if i == min(models)][0]
    data_train, data_test = utils.generate_test_train(data=data)

    # once we find best model then we predict and get residuals
    predictions = best_model.predict(data_test[core_features])
    residuals = np.array(data_test[target]) - predictions

    # generate all evals
    utils.generate_evals(target=target,
                         predictions=predictions,
                         data_test=data_test)