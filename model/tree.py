import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from src.data_utils import *

input_data, target_data = get_data(predict=False, horizon=1)

# train, validation, test 데이터 로더 생성
train_loader, valid_loader, test_loader, test_index = create_dataloaders(
    input_data, target_data, info, train_date, valid_date, test_date, batch_size)

# 데이터 로더를 사용하여 데이터 추출
def loader_to_dataframe(loader):
    inputs, outputs = [], []
    for batch in loader:
        x, c, y = batch
        inputs.append(x.numpy())
        outputs.append(y.numpy())
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    return inputs, outputs

train_inputs, train_outputs = loader_to_dataframe(train_loader)
valid_inputs, valid_outputs = loader_to_dataframe(valid_loader)
test_inputs, test_outputs = loader_to_dataframe(test_loader)

# 결정 트리 학습
tree_model = DecisionTreeRegressor(max_depth=4)
tree_model.fit(train_inputs, train_outputs)

# 검증 데이터로 평가
valid_predictions = tree_model.predict(valid_inputs)
mse = mean_squared_error(valid_outputs, valid_predictions)
print(f'Validation MSE for Decision Tree: {mse}')

# 테스트 데이터로 평가
test_predictions = tree_model.predict(test_inputs)
mse = mean_squared_error(test_outputs, test_predictions)
print(f'Test MSE for Decision Tree: {mse}')
