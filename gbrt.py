import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from data_utils import load_info, create_dataloaders, load_preprocessed_data

def extract_data_from_loader(data_loader):
    inputs, targets = [], []
    for batch_inputs, batch_targets in data_loader:
        inputs.append(batch_inputs.numpy())
        targets.append(batch_targets.numpy())
    inputs = np.vstack(inputs)
    targets = np.concatenate(targets)
    return inputs, targets

def main():
    input_data, target_data = load_preprocessed_data()
    print(input_data.shape, target_data.shape)
    firm_info, _ = load_info()

    train_loader, valid_loader, test_loader, _ = create_dataloaders(
        input_data, target_data, firm_info,
        train_date='2008-01-01', valid_date='2015-01-01', test_date='2023-11-01', batch_size=2000)

    print(len(train_loader), len(valid_loader), len(test_loader))

    x_train, y_train = extract_data_from_loader(train_loader)
    x_valid, y_valid = extract_data_from_loader(valid_loader)
    x_test, y_test = extract_data_from_loader(test_loader)

    # Combine train and valid data for GridSearchCV
    x_train_valid = np.vstack((x_train, x_valid))
    y_train_valid = np.concatenate((y_train, y_valid)).ravel()

    params = {
        "max_iter": [100, 200, 500],
        "max_depth": [1, 2],
        "learning_rate": [0.01, 0.05, 0.1]
    }

    gb_model = HistGradientBoostingRegressor()
    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(gb_model, param_grid=params, cv=tscv, scoring='neg_mean_squared_error', verbose=1, n_jobs=4)
    grid_search.fit(x_train_valid, y_train_valid)

    print(f"Best parameters: {grid_search.best_params_}")
    best_gb_model = grid_search.best_estimator_

    valid_mse = mean_squared_error(y_valid, best_gb_model.predict(x_valid))
    print(f"Validation MSE: {valid_mse}")

    gb_test_r2 = r2_score(y_test, best_gb_model.predict(x_test))
    print(f"Model Test R2: {gb_test_r2}")

if __name__ == '__main__':
    main()
