import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load in austin weather data for zipcode 78731 attained from NOAA
data = pd.read_csv('./data/weather/Austin - 78731.csv', parse_dates=['DATE'])

# generate numerical values from date column
data['YEAR'] = data.DATE.dt.year
data['MONTH'] = data.DATE.dt.month
data['DAY'] = data.DATE.dt.day

# core temperature features
target = 'TMAX'
core_features = ['PRCP', 'SNOW', 'SNWD', 'TMIN', 'YEAR', 'MONTH', 'DAY']

# filter only for core features
data_filtered = data[[target] + core_features]

# remove all na values
data_filtered = data_filtered.dropna()

# from documentation: 'Note: 9’s in a field (e.g.9999) indicate missing data or data that has not been received'
# remove these values as they are not valid to use in analysis

# correlation matrix
matrix = data_filtered.corr().round(2)
sns.heatmap(matrix, annot=True)
plt.show()

# split model into train and test data
X_train, X_test, y_train, y_test = train_test_split(data_filtered[core_features],
                                                    data_filtered[target],
                                                    test_size = 0.25)

# fit model with data
regr = LinearRegression()
regr.fit(X_train, y_train)

# get r-squared from test data
r_squared = regr.score(X_test, y_test)
print(r_squared)

# get predictions from model and get accuracy score
predictions = regr.predict(X_test)
