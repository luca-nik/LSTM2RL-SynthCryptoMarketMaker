import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import List, Dict
from datetime import datetime
import sys

class OrderBookGenerator(nn.Module):
    """
    A neural network model to generate the next best ask and best bid based on historical order book data.
    The model uses an LSTM to process sequences of order book data and predict the best ask and best bid.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """
        Initializes the OrderBookGenerator model.

        Args:
            input_dim (int): The number of features (dimensions) in the input order book data.
            hidden_dim (int): The number of features in the hidden layer of the LSTM.
        """
        super(OrderBookGenerator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 4)  # Output: best_ask, ask_price, best_bid, bid_price

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model to generate the next best ask and best bid.

        Args:
            x (torch.Tensor): The input tensor containing historical order book data (batch_size, seq_len, input_dim).
        
        Returns:
            torch.Tensor: The predicted next best ask and best bid.
        """
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use the last output for the next best_ask and best_bid

def prepare_data_for_training(orderbook_df: pd.DataFrame, seq_len: int):
    """
    Prepares sequences of orderbook data for training the LSTM.

    Args:
        orderbook_df (pd.DataFrame): The DataFrame with best ask, best bid, asks, and bids.
        seq_len (int): The length of the sequence for LSTM input.

    Returns:
        torch.Tensor: The input data as sequences.
        torch.Tensor: The target data (next best ask and best bid prices and quantities).
    """
    inputs = []
    targets = []

    # Iterate over the DataFrame to create sequences
    for i in range(len(orderbook_df) - seq_len):  # Ensure we have enough rows for sequences
        sequence = []
        for j in range(seq_len):
            # Extract asks and bids for the sequence
            asks = orderbook_df.iloc[i + j]['asks']
            bids = orderbook_df.iloc[i + j]['bids']

            # Append prices and quantities for asks and bids
            asks_bids = [float(asks[0]), float(asks[1]), float(bids[0]), float(bids[1])]
            sequence.append(asks_bids)

        # Target: best ask and bid prices and quantities for the next timestep
        best_ask_price = orderbook_df.iloc[i + seq_len]['asks'][0]
        best_ask_quantity = orderbook_df.iloc[i + seq_len]['asks'][1]
        best_bid_price = orderbook_df.iloc[i + seq_len]['bids'][0]
        best_bid_quantity = orderbook_df.iloc[i + seq_len]['bids'][1]
        target = [float(best_ask_price), float(best_ask_quantity), float(best_bid_price), float(best_bid_quantity)]

        inputs.append(sequence)
        targets.append(target)

    # Convert sequences and targets to tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)  # Shape: (num_sequences, seq_len, feature_size)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)  # Shape: (num_sequences, 4)

    return inputs_tensor, targets_tensor

def train_order_book_generator(model: nn.Module, orderbook_data: pd.DataFrame, seq_len: int = 5, epochs: int = 100, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Train the OrderBookGenerator model on historical order book data, predicting best_ask and best_bid with quantities.

    Args:
        model (nn.Module): The neural network model to be trained.
        orderbook_data (pd.DataFrame): A DataFrame containing order book states (best_ask, best_bid, asks, and bids).
        seq_len (int): The length of the input sequence for the LSTM.
        epochs (int): The number of training epochs (default is 100).
        device (torch.device): The device on which to perform the training (default is GPU if available, otherwise CPU).
    """
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare data and move to the appropriate device (GPU/CPU)
    inputs_tensor, targets_tensor = prepare_data_for_training(orderbook_data, seq_len)

    # Move data to the selected device (GPU/CPU)
    print(f'Offloading onto {device}')
    inputs_tensor = inputs_tensor.to(device)
    targets_tensor = targets_tensor.to(device)

    # Move model to the selected device
    model.to(device)

    for epoch in range(epochs):
        model.train()

        # Forward pass for each batch
        for i in range(len(inputs_tensor)):
            inputs = inputs_tensor[i].unsqueeze(0).to(device)  # Shape: (1, seq_len, feature_size)
            target = targets_tensor[i].unsqueeze(0).to(device)  # Shape: (1, 4)

            # Forward pass
            output = model(inputs)

            # Compute loss
            loss = criterion(output, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

#def generate_orderbook(model: nn.Module, initial_orderbook: List[List[List[float]]], num_steps: int) -> pd.DataFrame:
#    """
#    Generate a sequence of order book states using the trained model.
#
#    Args:
#        model (nn.Module): The trained neural network model.
#        initial_orderbook (List[List[List[float]]]): The initial sequence of order books (list of bids/asks).
#        num_steps (int): The number of steps to generate in the future.
#
#    Returns:
#        pd.DataFrame: A DataFrame containing the generated order books, with columns 'lastUpdated', 'asks', 'bids'.
#    """
#    generated_orderbooks = []
#    current_orderbook = initial_orderbook[-1]  # Start with the last available orderbook state
#
#    for step in range(num_steps):
#        # Generate the next order book state
#        orderbook_tensor = torch.tensor(current_orderbook, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
#        next_orderbook = model(orderbook_tensor).detach().numpy().flatten().tolist()
#
#        # Format the output into the expected structure (asks, bids)
#        # Assuming the output is flat and we need to restructure it into asks and bids
#        num_levels = len(next_orderbook) // 2  # Half for asks, half for bids
#        asks = [[next_orderbook[i], next_orderbook[i + num_levels]] for i in range(num_levels)]
#        bids = [[next_orderbook[i + num_levels], next_orderbook[i + num_levels + num_levels]] for i in range(num_levels)]
#        
#        # Get current timestamp
#        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.") + f"{int((datetime.now().microsecond) / 1000):03d}"
#        
#        generated_orderbooks.append({
#            'lastUpdated': timestamp,
#            'asks': asks,
#            'bids': bids
#        })
#
#        # Update the current_orderbook for the next iteration
#        current_orderbook = [asks, bids]
#
#    # Convert to DataFrame
#    orderbook_df = pd.DataFrame(generated_orderbooks)
#    return orderbook_df
def preprocess_orderbook_data(orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the orderbook data by extracting the best ask and best bid prices
    from the 'asks' and 'bids' columns for each time entry.

    Args:
        orderbook_df (pd.DataFrame): The full orderbook dataframe

    Returns:
        pd.DataFrame: The orderbook with only the best bid and best ask
    """
    best_asks = []
    best_bids = []
    last_updated = []

    for i, row in orderbook_df.iterrows():
        asks = row['asks']
        bids = row['bids']
        timestamp = row['lastUpdated']

        # Get the best ask (lowest ask price)
        best_ask = min(asks, key=lambda x: x[0])  # 'x[0]' is the price
        best_bid = max(bids, key=lambda x: x[0])  # 'x[0]' is the price
        
        best_asks.append([best_ask[0], best_ask[1]])  # Append the best ask price and quantity
        best_bids.append([best_bid[0], best_bid[1]])  # Append the best bid price and quantity
        last_updated.append(timestamp)
    
    # Create a new DataFrame with best_ask and best_bid
    processed_df = pd.DataFrame({
        'asks': best_asks,
        'bids': best_bids
    })
    
    return processed_df