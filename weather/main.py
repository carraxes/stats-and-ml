from weather.linear_regression import linear_regression

import pandas as pd

# from documentation: 'Note: 9’s in a field (e.g.9999) indicate missing data or data that has not been received'
# remove these values as they are not valid to use in analysis
# --- NOT AN ISSUE WHEN OUR FILTERING IS APPLIED ---

# load in austin weather data for zipcode 78731 attained from NOAA
data = pd.read_csv('./data/weather/Austin - 78731.csv', parse_dates=['DATE'])

if __name__ == '__main__':

    # run linear regression model on weather data
    linear_regression(data=data)