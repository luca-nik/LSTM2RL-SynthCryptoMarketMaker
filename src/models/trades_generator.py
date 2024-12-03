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


from utils.plotter import Plotter

class TradesGenerator(nn.Module):
    """
    A PyTorch-based LSTM model for generating order book data.

    Attributes:
        lstm (nn.LSTM): LSTM layer for sequential data processing.
        fc (nn.Linear): Fully connected layer for output predictions.
        features (List[str]): Feature names for the order book.
        device (torch.device, optional): Device for computation. Defaults to CPU.
        plotter (Plotter): Utility for plotting results and metrics.
        scaler (StandardScaler): Scaler for normalizing data.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int,  
                 CONFIG: dict, scaler: StandardScaler =  StandardScaler(), 
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        # LSTM
        super(TradesGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.features = ["Price", "Amount", "Side"]
        # Upload model on device
        self.device = device
        self.to(self.device)
        # Initialize Plotter
        self.plotter = Plotter(CONFIG['paths']['images_path'] + 'trades/', self.features)
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
        out[:, 2] = torch.relu(out[:, 1])  # Amount

        return out
    
    def train_model(self, train_loader: DataLoader, epochs: int, lr: float = 0.001, penalty_weight: float = 0.1) -> None:
        """
        Train the model on the given training data.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            epochs (int): Number of epochs to train.
            lr (float): Learning rate for the optimizer.
            device (torch.device): Device to run the training on.
            penalty_weight (float): Weight for volume penalty term.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        epoch_losses = []  # List to store average loss for each epoch
    
        for epoch in range(epochs):
            epoch_loss = 0.0  # Initialize epoch loss
    
            # Iterate through the batches in train_loader
            for trades_batch, target_batch in train_loader:
                trades_batch = trades_batch.to(self.device)
                target_batch = target_batch.to(self.device)
    
                # Forward pass
                output = self(trades_batch)
                
                # Compute MSE loss
                mse_loss = criterion(output, target_batch)
                
                # Compute penalty for negative predictions of ask and bid volumes
                negative_ask_volume = torch.sum(torch.clamp(output[:, 1], min=0))  # Penalty Amount
    
                # Total penalty
                penalty = penalty_weight * (negative_ask_volume)
    
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
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

        # Plot the loss history
        self.plotter.plot_loss(np.asarray(epoch_losses), target_directory = 'training/')
        
        # Evaluate trained model and plot
        predictions_original, targets_original = self.prediction(train_loader)
        
        # Plot Actual vs predicted on training set
        self.plotter.plot_actual_vs_predicted(targets_original, predictions_original, target_directory = 'training/')
    
    def prediction(self, data: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for a given dataset using the trained model.
    
        Args:
            data (DataLoader): A DataLoader object containing the input data and target values 
                               for generating predictions.
    
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - predictions_original (np.ndarray): Predicted values scaled back to the original scale.
                - targets_original (np.ndarray): Actual target values scaled back to the original scale.
        """
        self.eval()  # Set the model to evaluation mode
    
        predictions = []
        targets = []
    
        # Loop through the data loader to get the input and target data
        with torch.no_grad():
            for trades_batch, target_batch in data:
                trades_batch = trades_batch.to(self.device)  # Move input data to the model's device
                target_batch = target_batch.to(self.device)  # Move target data to the model's device
    
                # Get the predictions from the model
                predicted_values = self.predict(trades_batch)
    
                # Append predictions and targets as numpy arrays
                predictions.append(predicted_values.cpu().numpy())
                targets.append(target_batch.cpu().numpy())
    
        # Concatenate predictions and targets into numpy arrays for easier handling
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
    
        # Reverse the scaling transformation to get the original scale
        predictions_original = self.scaler.inverse_transform(predictions)
        targets_original = self.scaler.inverse_transform(targets)
        
        return predictions_original, targets_original

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

    def test(self, test_data: TensorDataset) -> np.ndarray:
        """
        Evaluate the model on the test dataset and visualize the results.
    
        Args:
            test_data (TensorDataset): Training dataset.
        
        Returns:
            predictions (np.ndarray): array of model predictions scaled to the original scale
        """
        print('  Testing the Trades model')

        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        predictions, targets = self.prediction(test_loader)

        # Print Test info
        rmse = np.sqrt(((predictions - targets) ** 2).mean())
        mae = np.abs(predictions - targets).mean()
        print(f'  RMSE: {rmse}')
        print(f'  MAE: {mae}')

        # Plot
        self.plotter.plot_actual_vs_predicted(targets, predictions, target_directory = 'test/')
        print(' ')

        return predictions

def trades_model_load_or_train(trades_train: TensorDataset, CONFIG: dict, retrain_model: bool = False,  
                               device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                               trades_scaler: StandardScaler =  StandardScaler()
                               ) -> TradesGenerator:
    """
    Load or train the TradesGenerator model.

    Args:
        trades_train (TensorDataset): Training dataset.
        CONFIG (dict): Configuration dictionary.
        retrain_model (bool, optional): Whether to retrain the model. Defaults to False.
        device (torch.device, optional): Device for computation. Defaults to CPU.
        trades_scaler (Optional[StandardScaler], optional): Scaler for normalizing data. Defaults to None.
    
    Returns:
        TradesGenerator: Trained model.
    """
    # Model setup
    input_size = trades_train.tensors[0].shape[-1]  # Number of features in each input sequence
    output_size = trades_train.tensors[0].shape[-1]   # Number of target features
    hidden_size = CONFIG['model']['hidden_size']  # Number of hidden units in LSTM
    num_layers = CONFIG['model']['num_layers']  # Single LSTM layer
    
    # Initialize model and load into device
    model = TradesGenerator(input_size, hidden_size, output_size, num_layers, CONFIG, trades_scaler, device)

    # Data loader
    trades_train_loader = DataLoader(trades_train, batch_size=CONFIG['training']['batch_size'], shuffle=False)

    # Path of previously trained model
    model_path = CONFIG['paths']['trades_model_path']

    # Check if the model exists and whether to retrain
    if os.path.exists(model_path) and not retrain_model:
        print("  Loading pretrained Trades model ...")
        model.load_state_dict(torch.load(model_path, weights_only=True))  # Explicitly set weights_only=True
        model.eval()
        print("  Trades model loaded!\n")
    else:
        print("  Training Trades model ...")
        # Train the model and get loss history
        epochs = CONFIG['training']['epochs']
        lr = CONFIG['training']['learning_rate']
        model.train_model(trades_train_loader, epochs=epochs, lr=lr, device=device)

        # Save the trained model
        torch.save(model.state_dict(), model_path)
        print("  Model saved!\n")


    return model


