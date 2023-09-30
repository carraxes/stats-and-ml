import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# from documentation: 'Note: 9’s in a field (e.g.9999) indicate missing data or data that has not been received'
# remove these values as they are not valid to use in analysis
# --- NOT AN ISSUE WHEN OUR FILTERING IS APPLIED ---

# load in austin weather data for zipcode 78731 attained from NOAA
data = pd.read_csv('./data/weather/Austin - 78731.csv', parse_dates=['DATE'])

# make index as date
data = data.sort_values(by=['DATE'])
data.index = data['DATE']

# shift TMAX value by one groupbed by STATION
TARGET = 'TMAX_NEXT_DAY'
data[TARGET] = data.groupby(['STATION']).TMAX.shift(-1)

# since there are multiple stations in the data, we would generate categorical variable based on STATION
# but since only one station actually has non-null TMAX values, we will not do this and instead
# remove null station data
data = data[pd.notnull(data.STATION)]

# core temperature features
core_features = ['PRCP', 'TMIN', 'TMAX']

# filter only for core features plus target
data_filtered = data[[TARGET] + core_features]

# remove all na values, while we could fill in the NULL percipitation values
# its a small % so we remove to facilitate analysis
data_filtered = data_filtered.dropna()

# Check linear relationship between features and target
for feature in core_features:
    plt.scatter(data_filtered[feature], data_filtered[TARGET])
    plt.show()

# Check for multicollinearity within data
X = add_constant(data_filtered[core_features])
vif_data = pd.Series([variance_inflation_factor(X.values, i) 
                      for i in range(X.shape[1])], index=X.columns)
print(vif_data)

# correlation matrix
matrix = data_filtered.corr().round(2)
sns.heatmap(matrix, annot=True)
plt.show()

# split model into train and test data
data_train = data_filtered[data_filtered.index < pd.to_datetime('6/1/2017')]
data_test = data_filtered[data_filtered.index >= pd.to_datetime('6/1/2017')]

# fit model with data
regr = LinearRegression()
regr.fit(data_train[core_features], data_train[TARGET])

# get predictions and generate residuals and plot
predictions = regr.predict(data_test[core_features])
residuals = np.array(data_test[TARGET]) - predictions
plt.scatter(predictions, residuals)
plt.show()

# get r-squared from test data
r_squared = regr.score(data_test[core_features], data_test[TARGET])
print(r_squared)

print(len(data_test[core_features]))
print(len(predictions))

# get our mae
mae = mean_absolute_error(data_test[TARGET], predictions)
print(mae)