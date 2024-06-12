import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging

from data_utils import load_info, create_dataloaders, load_preprocessed_data

logging.basicConfig(filename='model_output.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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


# def l1_regularization(model, l1_lambda):
#     l1_norm = sum(p.abs().sum() for p in model.parameters())
#     return l1_lambda * l1_norm

def calculate_r2_oos(predictions, actuals):
    numerator = np.sum((actuals - predictions) ** 2)
    denominator = np.sum(actuals ** 2)
    r2_oos = 1 - (numerator / denominator)
    return r2_oos



def train(train_loader, valid_loader, model, criterion, optimizer, epochs, patience):
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for data, target in train_loader:
            data = data.float().requires_grad_()
            target = target.float()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)# + l1_regularization(model, l1_lambda)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * data.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_valid_loss = 0.0
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.float()
                target = target.float()
                output = model(data)
                loss = criterion(output.squeeze(), target)
                running_valid_loss += loss.item() * data.size(0)

        epoch_valid_loss = running_valid_loss / len(valid_loader.dataset)

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_valid_loss:.4f}')

        # Early stopping
        if epoch_valid_loss < best_loss:
            best_loss = epoch_valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return best_loss

def hyperparameter_tuning(train_loader, valid_loader, input_dim, output_dim):
    param_grid = {
        'l1_penalty': np.logspace(-5, -3, num=3),
        'learning_rate': [0.001, 0.01],
        'batch_size': [1000, 2000, 3000]
    }

    best_params = None
    best_loss = float('inf')

    for params in ParameterGrid(param_grid):
        model = NeuralNetwork(input_dim=input_dim, hidden_layers=[32, 16, 8], output_dim=output_dim)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l1_penalty'])
        criterion = nn.MSELoss()

        loss = train(train_loader, valid_loader, model, criterion, optimizer, epochs=50, patience=5)

        if loss < best_loss:
            best_loss = loss
            best_params = params

    return best_params, best_loss

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.float()
            targets = targets.float()
            outputs = model(inputs)
            # print outputs by logging
            logging.info(f'outputs: {outputs}')
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())

    test_loss /= len(test_loader.dataset)
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    print(f'Test Loss: {test_loss:.4f}')

    r2_oos = calculate_r2_oos(predictions, actuals)
    print(f'R²_oos: {r2_oos:.4f}')


def main():
    input_data, target_data = load_preprocessed_data()
    print(input_data.shape, target_data.shape)
    firm_info, _ = load_info()

    train_loader, valid_loader, test_loader, test_index = create_dataloaders(
        input_data, target_data, firm_info,
        train_date='2008-01-01', valid_date='2017-01-01', test_date='2023-11-01', batch_size=1000)
    
    print(len(train_loader), len(valid_loader), len(test_loader))

    input_dim = input_data.shape[1] - 2
    output_dim = 1
    best_params, best_loss = hyperparameter_tuning(train_loader, valid_loader, input_dim, output_dim)

    print(f'Best Params: {best_params}')
    print(f'Best Validation Loss: {best_loss}')

    model = NeuralNetwork(input_dim=input_dim, hidden_layers=[32, 16, 8], output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['l1_penalty'])
    criterion = nn.MSELoss()

    print("Starting model training...")
    train(train_loader, valid_loader, model, criterion, optimizer, epochs=100, patience=5)

    print("Evaluating on test data...")
    test(test_loader, model, criterion)

if __name__ == "__main__":
    main()
