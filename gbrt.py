import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
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

    train_loader, valid_loader, test_loader, _ = create_dataloaders(
        input_data, target_data, firm_info,
        train_date='2008-01-01', valid_date='2015-01-01', test_date='2023-11-01', batch_size=2000)
    
    print(len(train_loader), len(valid_loader), len(test_loader))


    X_train, y_train = extract_data_from_loader(train_loader)
    X_valid, y_valid = extract_data_from_loader(valid_loader)
    X_test, y_test = extract_data_from_loader(test_loader)
    
    gb_model = HistGradientBoostingRegressor(
        max_iter=500,  # n_estimators
        learning_rate=0.01,
        max_depth=10,
        random_state=42,
        verbose=1  # 학습 진행 상황을 출력
    )
    gb_model.fit(X_train, y_train)

    gb_valid_preds = gb_model.predict(X_valid)
    gb_valid_acc = accuracy_score(y_valid, gb_valid_preds)

    print(f"Gradient Boosted Tree Validation Accuracy: {gb_valid_acc}")

    gb_test_preds = gb_model.predict(X_test)
    gb_test_acc = accuracy_score(y_test, gb_test_preds)

    print(f"Gradient Boosted Tree Test Accuracy: {gb_test_acc}")

if __name__ == '__main__':
    main()