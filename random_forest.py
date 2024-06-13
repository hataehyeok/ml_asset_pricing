import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

from data_utils import load_info, create_dataloaders, load_preprocessed_data

logging.basicConfig(filename='model_output.log', level=logging.INFO, format='%(asctime)s - %(message)s')


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

    train_loader, valid_loader, test_loader, test_index = create_dataloaders(
        input_data, target_data, firm_info,
        train_date='2008-01-01', valid_date='2015-01-01', test_date='2023-11-01', batch_size=2000)
    
    print(len(train_loader), len(valid_loader), len(test_loader))


    X_train, y_train = extract_data_from_loader(train_loader)
    X_valid, y_valid = extract_data_from_loader(valid_loader)
    X_test, y_test = extract_data_from_loader(test_loader)
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_valid_preds = rf_model.predict(X_valid)
    rf_valid_acc = accuracy_score(y_valid, rf_valid_preds)

    print(f"Random Forest Validation Accuracy: {rf_valid_acc}")

    rf_test_preds = rf_model.predict(X_test)
    rf_test_acc = accuracy_score(y_test, rf_test_preds)

    print(f"Random Forest Test Accuracy: {rf_test_acc}")    

if __name__ == '__main__':
    main()