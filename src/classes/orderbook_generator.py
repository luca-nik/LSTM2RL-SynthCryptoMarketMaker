import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


from classes.plotter import Plotter

class OrderBookGenerator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int,  CONFIG: dict, scaler: StandardScaler =  StandardScaler()):
        super(OrderBookGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.features = ["Best Ask", "Ask Volume", "Best Bid", "Bid Volume"]
        # Initialize Plotter
        self.plotter = Plotter(CONFIG['paths']['images_path'] + 'orderbook/', self.features)
        self.scaler = scaler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Check if lstm_out is 3D
        if lstm_out.dim() == 3:
            # If lstm_out is 3D, take the output of the last time step
            out = self.fc(lstm_out[:, -1, :])
        elif lstm_out.dim() == 2:
            # If lstm_out is 2D, directly use it (you might need to adjust based on your setup)
            out = self.fc(lstm_out)
        else:
            raise ValueError("Unexpected output shape from LSTM")
        #out = self.fc(lstm_out[:, -1, :])  # Take the output of the last time step

        # Apply ReLU to ask volume (index 1) and bid volume (index 3)
        out[:, 1] = torch.relu(out[:, 1])  # Ask volume
        out[:, 3] = torch.relu(out[:, 3])  # Bid volume

        return out

    def train_model(self, train_loader: DataLoader, epochs: int, lr: float = 0.001, device: torch.device = torch.device('cpu'), penalty_weight: float = 0.1) -> list:
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
                
                # Compute MSE loss
                mse_loss = criterion(output, target_batch)
                
                # Compute penalty for negative predictions of ask and bid volumes
                negative_ask_volume = torch.sum(torch.clamp(output[:, 1], min=0))  # Penalty for negative ask volume
                negative_bid_volume = torch.sum(torch.clamp(output[:, 3], min=0))  # Penalty for negative bid volume
    
                # Total penalty
                penalty = penalty_weight * (negative_ask_volume + negative_bid_volume)
    
                # Total loss (MSE loss + penalty)
                total_loss = mse_loss + penalty
    
                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Accumulate loss for this batch
                epoch_loss += total_loss.item()
    
            # Compute average loss for this epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            epoch_losses.append(avg_epoch_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

        # Plot the loss history
        self.plotter.plot_loss(np.asarray(epoch_losses), target_directory = 'training/')
        
        # Evaluate trained model and plot
        self.eval()

        predictions = []
        targets = []

        # Loop through the train_loader to get the data
        with torch.no_grad():
            for orderbook_batch, target_batch in train_loader:
                orderbook_batch = orderbook_batch.to(device)
                target_batch = target_batch.to(device)

                # Get the predictions from the self
                predicted_values = self.predict(orderbook_batch)

                # Store the predictions and actual targets
                predictions.append(predicted_values.cpu().numpy())
                targets.append(target_batch.cpu().numpy())

        # Convert predictions and targets into a numpy array for easier plotting
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Inverse-transform data to the original scale using the scaler
        predictions_original = self.scaler.inverse_transform(predictions)
        targets_original = self.scaler.inverse_transform(targets)
        
        # Plot Actual vs predicted on training set
        self.plotter.plot_actual_vs_predicted(targets_original, predictions_original, target_directory = 'training/')

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

    def test(self, test_data, device: torch.device):
        """
        Evaluate the model on the test dataset and visualize the results.
    
        Args:
            test_data
            device (torch.device): Device (CPU or GPU) where the model runs.
        """
        self.eval()  # Set the model to evaluation mode

        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        predictions = []
        targets = []

        # Loop through the train_loader to get the data
        with torch.no_grad():
            for orderbook_batch, target_batch in test_loader:
                orderbook_batch = orderbook_batch.to(device)
                target_batch = target_batch.to(device)

                # Get the predictions from the model
                predicted_values = self.predict(orderbook_batch)

                # Store the predictions and actual targets
                predictions.append(predicted_values.cpu().numpy())
                targets.append(target_batch.cpu().numpy())

        # Convert predictions and targets into a numpy array for easier plotting
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Inverse-transform data to the original scale using the scaler
        predictions_original = self.scaler.inverse_transform(predictions)
        targets_original = self.scaler.inverse_transform(targets)

        # Print Test info
        rmse = np.sqrt(((predictions - targets) ** 2).mean())
        mae = np.abs(predictions - targets).mean()
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')

        # Plot
        self.plotter.plot_actual_vs_predicted(targets_original, predictions_original, target_directory = 'test/')
        return predictions
        
def orderbook_model_load_or_train(orderbook_train: TensorDataset, CONFIG: dict, retrain_model: bool = False, \
                                  device : torch.device = torch.device('cpu'), orderbook_scaler: StandardScaler =  StandardScaler()):
    

    # Model setup
    input_size = orderbook_train.tensors[0].shape[-1]  # Number of features in each input sequence
    output_size = orderbook_train.tensors[0].shape[-1]   # Number of target features
    hidden_size = CONFIG['model']['hidden_size']  # Number of hidden units in LSTM
    num_layers = CONFIG['model']['num_layers']  # Single LSTM layer
    
    # Initialize model and load into device
    model = OrderBookGenerator(input_size, hidden_size, output_size, num_layers, CONFIG, orderbook_scaler)
    model.to(device)

    # Data loader
    orderbook_train_loader = DataLoader(orderbook_train, batch_size=CONFIG['training']['batch_size'], shuffle=False)

    # Path of previously trained model
    model_path = CONFIG['paths']['model_path']

    # Check if the model exists and whether to retrain
    if os.path.exists(model_path) and not retrain_model:
        print("Loading pretrained model ...")
        model.load_state_dict(torch.load(model_path, weights_only=True))  # Explicitly set weights_only=True
        model.eval()
    else:
        print("Training model ...")
        # Train the model and get loss history
        epochs = CONFIG['training']['epochs']
        lr = CONFIG['training']['learning_rate']
        model.train_model(orderbook_train_loader, epochs=epochs, lr=lr, device=device)

        # Save the trained model
        torch.save(model.state_dict(), model_path)
        print("Model saved!\n")


    return model


