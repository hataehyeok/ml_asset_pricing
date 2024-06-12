import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from data_utils import load_info, create_dataloaders, load_preprocessed_data

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

def l1_regularization(model, l1_lambda):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_lambda * l1_norm

def calculate_r2_oos(predictions, actuals):
    numerator = np.sum((actuals - predictions) ** 2)
    denominator = np.sum(actuals ** 2)
    r2_oos = 1 - (numerator / denominator)
    return r2_oos

def train(model, train_loader, valid_loader, criterion, optimizer, epochs, patience, l1_lambda, best_loss, patience_counter):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target) + l1_regularization(model, l1_lambda)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data)
                loss = criterion(output.squeeze(), target)
                val_loss += loss.item() * data.size(0)
        
        val_loss /= len(valid_loader.dataset)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
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
        train_date='1993-12-01', valid_date='2010-01-01', test_date='2018-01-01', batch_size=1000)
    
    # Hyperparameters setting
    input_dim = input_data.shape[1] - 2
    output_dim = 1
    learning_rate = 0.001
    epochs = 100
    patience = 5
    l1_lambda = 1e-5

    model = NeuralNetwork(input_dim, [128, 64, 32], output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    best_loss = float('inf')
    patience_counter = 0

    train(model, train_loader, valid_loader, criterion, optimizer, epochs, patience, l1_lambda, best_loss, patience_counter)
    test(model, test_loader, criterion)
    

if __name__ == '__main__':
    main()