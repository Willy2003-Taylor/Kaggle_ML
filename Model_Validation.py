import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Build a model #

melbourne_file_path = '../melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)
filtered_melbourne_data = melbourne_data.dropna(axis = 0)
"""
Filter rows with missing price values, axis = 0 stands for rows
axis = 1 stands for columns
"""

y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']

X = filtered_melbourne_data[melbourne_features]

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(X, y)

# Calculate Mean Absolute Error #

predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))

# Test Validation Data #

train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state = 0)

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

validation_predictions = melbourne_model.predict(validation_X)
print(mean_absolute_error(validation_y, validation_predictions))
