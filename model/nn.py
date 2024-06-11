import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from src.data_utils import *


#################################################################################################################################

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

def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, concepts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, concepts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

def validate_model(model, valid_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, concepts, labels in valid_loader:
            outputs = model(inputs, concepts)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(valid_loader.dataset)
    print(f'Validation Loss: {epoch_loss:.4f}')
    return epoch_loss

def test_model(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, concepts, labels in test_loader:
            outputs = model(inputs, concepts)
            predictions.append(outputs.numpy())
    return predictions

def main():
    input_data, target_data = get_data()
    firm_info, _ = load_info()

    train_loader, valid_loader, test_loader, test_index = create_dataloaders(
        input_data, target_data, firm_info, 
        train_date='2015-01-01', valid_date='2017-01-01', test_date='2019-01-01', batch_size=64
    )

    input_dim = input_data.shape[1] - len(firm_info[firm_info['Cat.Data'] == 'Analyst']['Acronym'].values) - 2
    concept_dim = len(firm_info[firm_info['Cat.Data'] == 'Analyst']['Acronym'].values)
    output_dim = 1
    model = NeuralNetwork(input_dim, concept_dim, output_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 25
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    validate_model(model, valid_loader, criterion)

    predictions = test_model(model, test_loader)
    print("Test Predictions: ", predictions)

if __name__ == "__main__":
    main()