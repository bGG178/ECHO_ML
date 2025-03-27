import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Example Dataset class
class ECVTDataset(Dataset):
    def __init__(self, capacitance_data, area_labels):
        """
        :param capacitance_data: NumPy array of shape (num_samples, num_sensors)
        :param area_labels: NumPy array of shape (num_samples, 1)
        """
        self.capacitance_data = torch.tensor(capacitance_data, dtype=torch.float32)
        self.area_labels = torch.tensor(area_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.capacitance_data)

    def __getitem__(self, idx):
        return self.capacitance_data[idx], self.area_labels[idx]

# Define the model
class ECVTNet(nn.Module):
    def __init__(self, input_size):
        super(ECVTNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output is a single float (predicted area)
        )

    def forward(self, x):
        return self.fc(x)

# Training function
def train_model(model, train_loader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for capacitance, area in train_loader:
            optimizer.zero_grad()
            predictions = model(capacitance)
            loss = criterion(predictions, area)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

# Generate synthetic training data
num_samples = 1000
num_sensors = 16  # Example: 16 capacitance sensors
capacitance_data = np.random.rand(num_samples, num_sensors)  # Random capacitance values
true_areas = np.sum(capacitance_data, axis=1, keepdims=True) / num_sensors * 100  # Example relationship

# Prepare dataset and model
dataset = ECVTDataset(capacitance_data, true_areas)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = ECVTNet(input_size=num_sensors)

# Train the model
train_model(model, train_loader)
