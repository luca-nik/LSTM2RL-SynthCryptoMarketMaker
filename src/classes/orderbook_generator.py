import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import Tuple

class OrderBookGenerator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        """
        Initialize the OrderBookGenerator LSTM model.

        Args:
            input_size (int): Number of input features (e.g., 4 for best ask, best ask volume, best bid, best bid volume).
            hidden_size (int): Number of hidden units in the LSTM layer.
            output_size (int): Number of output features (e.g., 4 for next best ask, next best ask volume, best bid, best bid volume).
            num_layers (int): Number of LSTM layers (default is 1).
        """
        super(OrderBookGenerator, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to predict the next best ask, ask volume, best bid, and bid volume
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input sequence tensor with shape (batch_size, sequence_length, input_size).
        
        Returns:
            torch.Tensor: Predicted next best ask, best ask volume, best bid, best bid volume.
        """
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(x)  # h_n is the hidden state
        # Use the last hidden state for prediction (h_n[-1] corresponds to the last time step)
        out = self.fc(lstm_out[:, -1, :])  # Take the output of the last time step
        return out

    def train_model(self, train_data: Tuple[torch.Tensor, torch.Tensor], epochs: int, lr: float = 0.001):
        """
        Train the model using LSTM.

        Args:
            train_data (tuple): A tuple containing (input_data, target_data).
            epochs (int): Number of epochs to train.
            lr (float): Learning rate for the optimizer (default is 0.001).
        """
        input_data, target_data = train_data
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            # Forward pass
            output = self(input_data)
            
            # Compute loss
            loss = criterion(output, target_data)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
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