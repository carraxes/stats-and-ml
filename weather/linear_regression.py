import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

def _generate_linear_comparisons(target: str, core_features: list, data_filtered: pd.DataFrame) -> None:
    """check linear relationship between features and target"""
    for feature in core_features:
        plt.scatter(data_filtered[feature], data_filtered[TARGET])
        plt.show()

def _generate_vif_test(core_features: list, data_filtered: pd.DataFrame) -> None:
    """check for multicollinearity within data"""
    X = add_constant(data_filtered[core_features])
    vif_data = pd.Series([variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])], index=X.columns)
    print(vif_data)

def _generate_correlation_matrix(data_filtered: pd.DataFrame) -> None:
    """correlation matrix"""
    matrix = data_filtered.corr().round(2)
    sns.heatmap(matrix, annot=True)
    plt.show()

def _generate_residual_plot(target: str, predictions: list, data_test: pd.DataFrame) -> None:
    """residual plot"""
    residuals = np.array(data_test[target]) - predictions
    plt.scatter(predictions, residuals)
    plt.show()

def linear_regression(data: pd.DataFrame,
                      do_vif_test: bool = False,
                      do_residuals: bool = False,
                      do_linear_comparison: bool = False,
                      do_correlation_matrix: bool = False) -> None:
    """Linear regression model on weather data from NOAA."""

    # make index as date
    data = data.sort_values(by=['DATE'])
    data.index = data['DATE']

    # shift TMAX value by one groupbed by STATION
    TARGET = 'TMAX_NEXT_DAY'
    data[TARGET] = data.groupby(['STATION']).TMAX.shift(-1)

    core_features = ['PRCP', 'TMIN', 'TMAX']
    # since there are multiple stations in the data, we would generate categorical variable based on STATION
    # but since only one station actually has non-null TMAX values, we will not do this and instead
    # remove null station data
    data = data[[TARGET] + core_features].dropna()

    # generate additional features
    for col in ['TMIN', 'TMAX']:
        col_30_day_mean = f'{col}_30_DAY'
        data[col_30_day_mean] = data[col].rolling(30).mean()
        core_features.append(col_30_day_mean)

    # filter only for core features plus target
    data_filtered = data[[TARGET] + core_features]

    # remove all na values, while we could fill in the NULL percipitation values
    # its a small % so we remove to facilitate analysis
    data_filtered = data_filtered.dropna()

    # check linear relationship between features and target
    if do_linear_comparison:
        _generate_linear_comparisons(target=TARGET,
                                     core_features=core_features,
                                     data_filtered=data_filtered)

    # check for multicollinearity within data
    if do_vif_test:
        _generate_vif_test(core_features=core_features, data_filtered=data_filtered)

    # correlation matrix
    if do_correlation_matrix:
        _generate_correlation_matrix(data_filtered=data_filtered)

    # split model into train and test data
    data_train = data_filtered[data_filtered.index < pd.to_datetime('6/1/2017')]
    data_test = data_filtered[data_filtered.index >= pd.to_datetime('6/1/2017')]

    # fit model with data
    regr = LinearRegression()
    regr.fit(data_train[core_features], data_train[TARGET])
    predictions = regr.predict(data_test[core_features])

    # get predictions and generate residuals and plot
    if do_residuals:
        _generate_residual_plot(target=TARGET, predictions=predictions)

    # get r-squared from test data
    r_squared = regr.score(data_test[core_features], data_test[TARGET])
    print(f'R-Squared (Linear Regression): {r_squared}')

    # get the mae
    mae = mean_absolute_error(data_test[TARGET], predictions)
    print(f'Mean Absolute Error (Linear Regression): {mae}')