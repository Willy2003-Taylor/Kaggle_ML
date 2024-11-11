from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

def get_mean_absolute_error(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    pred_validation = model.predict(val_X)
    mae = mean_absolute_error(val_y, pred_validation)
    return mae

melbourne_file_path = '../melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

filtered_melbourne_data = melbourne_data.dropna(axis = 0)

y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']

X = filtered_melbourne_data[melbourne_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mean_absolute_error(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d Mean Absolute Error: %f" % (max_leaf_nodes, my_mae))

