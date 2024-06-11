import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from data_utils import load_info, create_dataloaders

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_and_evaluate_model(train_loader, valid_loader, model, criterion, optimizer, epochs, patience):
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                valid_loss += criterion(outputs, targets).item()
        valid_loss /= len(valid_loader)

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_loss

def hyperparameter_tuning(train_loader, valid_loader, input_dim, output_dim):
    param_grid = {
        'l1_penalty': np.logspace(-5, -3, num=3),
        'learning_rate': [0.001, 0.01],
        'batch_size': [10000]
    }

    best_params = None
    best_loss = float('inf')

    for params in ParameterGrid(param_grid):
        model = NeuralNetwork(input_dim=input_dim, hidden_layers=[32, 16, 8], output_dim=output_dim)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l1_penalty'])
        criterion = nn.MSELoss()

        loss = train_and_evaluate_model(train_loader, valid_loader, model, criterion, optimizer, epochs=100, patience=5)

        if loss < best_loss:
            best_loss = loss
            best_params = params

    return best_params, best_loss


def load_preprocessed_data():
    input_data = pd.read_csv('../data/preprocessed/input.csv')
    target_data = pd.read_csv('../data/preprocessed/target.csv')
    
    input_data['date'] = pd.to_datetime(input_data['date'])
    target_data['date'] = pd.to_datetime(target_data['date'])
    
    return input_data, target_data

def main():
    input_data, target_data = load_preprocessed_data()
    firm_info, _ = load_info()

    train_loader, valid_loader, test_loader, test_index = create_dataloaders(
        input_data, target_data, 
        train_date='1988-01-01', valid_date='2000-01-01', test_date='2006-01-01', batch_size=2000
    )

    input_dim = input_data.shape[1]- 2
    output_dim = 1

    best_params, best_loss = hyperparameter_tuning(train_loader, valid_loader, input_dim, output_dim)

    print(f'Best Params: {best_params}')
    print(f'Best Validation Loss: {best_loss}')

if __name__ == "__main__":
    main()
