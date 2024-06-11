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
    def __init__(self, input_dim, concept_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128 + concept_dim, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x, c):
        x = torch.relu(self.fc1(x))
        x = torch.cat((x, c), dim=1)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, train_loader, criterion, optimizer, concept_dim, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs[:, :-concept_dim], inputs[:, -concept_dim:])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

def validate_model(model, valid_loader, criterion, concept_dim):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            outputs = model(inputs[:, :-concept_dim], inputs[:, -concept_dim:])
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(valid_loader.dataset)
    print(f'Validation Loss: {epoch_loss:.4f}')
    return epoch_loss

def test_model(model, test_loader, concept_dim):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs[:, :-concept_dim], inputs[:, -concept_dim:])
            predictions.append(outputs.numpy())
    return predictions

def hyperparameter_tuning(train_loader, valid_loader, input_dim, concept_dim, output_dim):
    param_grid = {
        'l1_penalty': np.logspace(-5, -3, num=3),
        'learning_rate': [0.001, 0.01],
        'batch_size': [10000]
    }

    best_params = None
    best_loss = float('inf')

    for params in ParameterGrid(param_grid):
        model = NeuralNetwork(input_dim=input_dim, concept_dim=concept_dim, output_dim=output_dim)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l1_penalty'])
        criterion = nn.MSELoss()

        loss = train_and_evaluate_model(train_loader, valid_loader, model, criterion, optimizer, concept_dim, epochs=100, patience=5)

        if loss < best_loss:
            best_loss = loss
            best_params = params

    return best_params, best_loss

def train_and_evaluate_model(train_loader, valid_loader, model, criterion, optimizer, concept_dim, epochs, patience):
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs[:, :-concept_dim], inputs[:, -concept_dim:])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs[:, :-concept_dim], inputs[:, -concept_dim:])
                valid_loss += criterion(outputs, labels).item()
        valid_loss /= len(valid_loader)

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_loss


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
        train_date='1960-01-01', valid_date='1995-01-01', test_date='2006-01-01', batch_size=5000
    )

    input_dim = input_data.shape[1] - len(firm_info[firm_info['Cat.Data'] == 'Analyst']['Acronym'].values) - 2
    concept_dim = len(firm_info[firm_info['Cat.Data'] == 'Analyst']['Acronym'].values)
    output_dim = 1

    best_params, best_loss = hyperparameter_tuning(train_loader, valid_loader, input_dim, concept_dim, output_dim)

    print(f'Best Params: {best_params}')
    print(f'Best Validation Loss: {best_loss}')

    model = NeuralNetwork(input_dim, concept_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['l1_penalty'])
    criterion = nn.MSELoss()

    train_model(model, train_loader, criterion, optimizer, concept_dim, num_epochs=100)
    validate_model(model, valid_loader, criterion, concept_dim)

    predictions = test_model(model, test_loader, concept_dim)
    print("Test Predictions: ", predictions)

if __name__ == "__main__":
    main()
