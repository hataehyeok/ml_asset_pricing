import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 데이터 생성 (여기서는 임의의 데이터로 대체)
np.random.seed(0)
X_train = np.random.rand(100000, 20).astype(np.float32)
y_train = np.random.randint(2, size=100000).astype(np.float32)
X_val = np.random.rand(20000, 20).astype(np.float32)
y_val = np.random.randint(2, size=20000).astype(np.float32)

# 데이터셋 및 데이터로더 생성
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

batch_size = 10000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# L1 정규화 함수
def l1_regularization(model, l1_lambda):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_lambda * l1_norm

# 하이퍼파라미터 설정
input_dim = X_train.shape[1]
learning_rate = 0.001
epochs = 100
patience = 5
l1_lambda = 1e-5

# 모델 초기화
model = SimpleNN(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 얼리스탑핑 설정
best_loss = float('inf')
patience_counter = 0

# 학습 루프
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
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output.squeeze(), target)
            val_loss += loss.item() * data.size(0)
    
    val_loss /= len(val_loader.dataset)
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # 얼리스탑핑 체크
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        # 모델 저장 (옵션)
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# 최종 모델 로드 (옵션)
model.load_state_dict(torch.load('best_model.pth'))
