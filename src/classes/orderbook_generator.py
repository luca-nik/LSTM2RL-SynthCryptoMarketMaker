import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class OrderBookGenerator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super(OrderBookGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take the output of the last time step
        return out

    def train_model(self, train_loader: DataLoader, epochs: int, lr: float = 0.001, device: torch.device = torch.device('cpu')) -> list:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        epoch_losses = []  # List to store average loss for each epoch

        for epoch in range(epochs):
            epoch_loss = 0.0  # Initialize epoch loss

            # Iterate through the batches in train_loader
            for orderbook_batch, target_batch in train_loader:
                orderbook_batch = orderbook_batch.to(device)
                target_batch = target_batch.to(device)

                # Forward pass
                output = self(orderbook_batch)
                
                # Compute loss
                loss = criterion(output, target_batch)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate loss for this batch
                epoch_loss += loss.item()

            # Compute average loss for this epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            epoch_losses.append(avg_epoch_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")
        
        return epoch_losses  # Return the loss history for plotting  

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Predict the next order book values (best ask, ask volume, best bid, bid volume).

        Args:
            input_data (torch.Tensor): Input sequence tensor with shape (batch_size, sequence_length, input_size).
        
        Returns:
            torch.Tensor: Predicted next best ask, ask volume, best bid, and bid volume.
        """
        with torch.no_grad():
            return self(input_data)