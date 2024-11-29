import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


from classes.plotter import Plotter

class OrderBookGenerator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super(OrderBookGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take the output of the last time step

        # Apply ReLU to ask volume (index 1) and bid volume (index 3)
        out[:, 1] = torch.relu(out[:, 1])  # Ask volume
        out[:, 3] = torch.relu(out[:, 3])  # Bid volume

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
    
def orderbook_model_load_or_train(orderbook_train: TensorDataset, CONFIG: dict, retrain_model: bool = False, \
                                  device : torch.device = torch.device('cpu'), orderbook_scaler: StandardScaler =  StandardScaler()):
    
    # Initialize Plotter
    plotter = Plotter(CONFIG['paths']['images_path'])

    # Model setup
    input_size = orderbook_train.tensors[0].shape[-1]  # Number of features in each input sequence
    output_size = orderbook_train.tensors[0].shape[-1]   # Number of target features
    hidden_size = CONFIG['model']['hidden_size']  # Number of hidden units in LSTM
    num_layers = CONFIG['model']['num_layers']  # Single LSTM layer
    
    # Initialize model and load into device
    model = OrderBookGenerator(input_size, hidden_size, output_size, num_layers)
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
        epoch_losses = model.train_model(orderbook_train_loader, epochs=epochs, lr=lr, device=device)

        # Save the trained model
        torch.save(model.state_dict(), model_path)
        print("Model saved!")

        # Plot the loss history
        plotter.plot_loss(epoch_losses)


    # Use the plotter to save plots after the model evaluation
    plotter.plot_actual_vs_predicted(model, orderbook_train_loader, device, orderbook_scaler)
