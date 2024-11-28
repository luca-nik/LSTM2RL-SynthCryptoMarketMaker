import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

from helpers.data_loader import read_data
from helpers.data_handler import preprocess_and_split_data, create_sequences, standardize_data
from helpers.configurator import load_config
from classes.orderbook_generator import OrderBookGenerator
from classes.plotter import Plotter

# Load config file
CONFIG = load_config()

# Define retraining flag
retrain_model = False  # Set to True if you want to retrain the model

# Set pandas display option to show 8 digits after the decimal point
pd.set_option('display.float_format', '{:.8f}'.format)

# Initialize Plotter
plotter = Plotter(CONFIG['paths']['images_path'])

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Print a message indicating that data loading is starting
print("Loading orderbook and trading data ...")

# Load the orderbook and trading data
orderbook_df, trades_df = read_data(CONFIG)

# Print a message indicating that the loading process is completed
print("Loading completed ...\n")

# Preprocess the data and split them into train test
orderbook_train, trades_train, orderbook_test, trades_test = preprocess_and_split_data(orderbook_df, trades_df, CONFIG)

# Step 1: Standardize the Data
orderbook_train_scaled, orderbook_test_scaled = standardize_data(orderbook_train, orderbook_test)
trades_train_scaled, trades_test_scaled = standardize_data(trades_train, trades_test)

# Step 2: Create Sequences
orderbook_sequences_scaled, trades_sequences_scaled, target_sequences_scaled = create_sequences(
    orderbook_train_scaled, trades_train_scaled, CONFIG)

# The sequences are already tensors, so no need to use np.concatenate
# Just directly use the tensors as they are returned from create_sequences
orderbook_sequences_scaled_tensor = orderbook_sequences_scaled
target_sequences_scaled_tensor = target_sequences_scaled

# Step 3: Prepare DataLoader for Training
train_dataset = TensorDataset(orderbook_sequences_scaled_tensor, target_sequences_scaled_tensor)
sys.exit()

# Using batch_size=64 without shuffling, since this is time-series data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

input_size = orderbook_sequences_scaled.shape[2]  # Number of features in each input sequence
hidden_size = 64  # Number of hidden units in LSTM
output_size = target_sequences_scaled.shape[1]  # Number of target features
num_layers = 1  # Single LSTM layer

# Initialize the model
model = OrderBookGenerator(input_size, hidden_size, output_size, num_layers)
model.to(device)

# Define the model path
model_path = "data/orderbook_generator.pth"

# Check if the model exists and whether to retrain
if os.path.exists(model_path) and not retrain_model:
    print("Loading pretrained model ...")
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Explicitly set weights_only=True
    model.eval()
else:
    print("Training model ...")
    # Train the model and get loss history
    epochs = 100
    epoch_losses = model.train_model(train_loader, epochs=epochs, lr=0.005, device=device)

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print("Model saved!")

    # Plot the loss history
    plotter.plot_loss(epoch_losses)


# Use the plotter to save plots after the model evaluation
plotter.plot_actual_vs_predicted(model, train_loader, device)