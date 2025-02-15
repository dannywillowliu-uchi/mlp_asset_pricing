import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
#testing 

# Define the dataset
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

# Define the dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, sequence_length):
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.sequence_length]
        y = self.targets[index + self.sequence_length - 1]
        return x, y

# Define the MLP model
class MLP(nn.Module):
    #layer number alterable, maybe test varying layer#
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No ReLU after the last layer
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for data, targets in dataloader:
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example usage
    # Example data
    data = torch.randn(100, 10)  # 100 samples, 10 features each
    targets = torch.randn(100, 1)  # 100 targets

    # Hyperparameters
    input_size = 10
    hidden_size = 50
    output_size = 1
    num_epochs = 20
    batch_size = 16
    learning_rate = 0.001
    sequence_length = 5  # Length of the input sequence for time-series prediction
    layer_sizes = [input_size * sequence_length, 50, 30, output_size]  # Adjust input size for sequences

    # Dataset and DataLoader
    dataset = TimeSeriesDataset(data, targets, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, criterion, and optimizer
    model = MLP(layer_sizes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs)
