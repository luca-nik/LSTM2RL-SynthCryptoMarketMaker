import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Tuple, Dict

class TradesGenerator(nn.Module):
    """
    A neural network model that generates market trades based on the order book data.
    
    The model predicts the trade price, amount, and side (buy or sell) given the features of the order book.
    
    Args:
        input_dim (int): The number of input features (e.g., best bid price, best ask price, etc.).
        hidden_dim (int): The number of hidden units in the neural network.
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(TradesGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output: trade price
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output: trade amount
        self.fc4 = nn.Linear(hidden_dim, 2)  # Output: binary classification for side (buy or sell)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the neural network to generate predictions for price, amount, and side.
        
        Args:
            x (torch.Tensor): The input tensor containing order book features.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - predicted price (tensor)
                - predicted amount (tensor)
                - predicted side (tensor)
        """
        x = torch.relu(self.fc1(x))
        price = self.fc2(x)  # Predicted price
        amount = self.fc3(x)  # Predicted amount
        side = self.fc4(x)  # Predicted side (buy or sell)
        return price, amount, side

# Function to train the TradesGenerator
def train_trades_generator(model: nn.Module, orderbook_data: pd.DataFrame, epochs: int = 100) -> None:
    criterion_price = nn.MSELoss()
    criterion_amount = nn.MSELoss()
    criterion_side = nn.CrossEntropyLoss()  # Binary classification for side
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()

        for idx, orderbook in orderbook_data.iterrows():
            # Prepare input features for the model
            features = np.array([
                orderbook['bids'][0][0],  # Best bid price
                orderbook['asks'][0][0],  # Best ask price
                orderbook['bids'][0][1],  # Best bid quantity
                orderbook['asks'][0][1],  # Best ask quantity
                (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2  # Fair price (mid)
            ])

            features_tensor = torch.tensor(features, dtype=torch.float32)

            # Generate trade price, amount, and side
            price, amount, side = model(features_tensor)

            # Simulate ground truth labels for price, amount, and side (using random values as placeholder)
            target_price = torch.tensor([orderbook['bids'][0][0]], dtype=torch.float32)
            target_amount = torch.tensor([orderbook['bids'][0][1]], dtype=torch.float32)
            target_side = torch.tensor([0 if orderbook['bids'][0][0] > orderbook['asks'][0][0] else 1], dtype=torch.long)

            # Compute losses
            loss_price = criterion_price(price, target_price)
            loss_amount = criterion_amount(amount, target_amount)
            loss_side = criterion_side(side, target_side)

            # Total loss
            total_loss = loss_price + loss_amount + loss_side

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss.item()}")

def generate_trades(model: nn.Module, orderbook_data: pd.DataFrame, num_trades: int = 100) -> pd.DataFrame:
    """
    Generate a set of trades based on the predicted output from the trained model and the provided orderbook data.
    
    Args:
        model (nn.Module): The trained neural network model that generates trade predictions.
        orderbook_data (pd.DataFrame): The order book data containing bids and asks.
        num_trades (int): The number of trades to generate (default is 100).
    
    Returns:
        pd.DataFrame: A DataFrame containing the generated trades with columns: timestamp, id, price, side, amount.
    """
    trades = []
    
    # Ensure that we do not exceed the number of available rows in orderbook_data
    for idx in range(min(num_trades, len(orderbook_data))):  # Modify loop range to avoid out-of-bounds access
        orderbook = orderbook_data.iloc[idx]  # Get the orderbook at time idx
        
        # Prepare input features for the model
        features = np.array([
            orderbook['bids'][0][0],  # Best bid price
            orderbook['asks'][0][0],  # Best ask price
            orderbook['bids'][0][1],  # Best bid quantity
            orderbook['asks'][0][1],  # Best ask quantity
            (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2  # Fair price (mid)
        ])
        
        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Generate trade price, amount, and side
        price, amount, side = model(features_tensor)
        
        # Apply sigmoid activation to side and get the index of the higher value (buy vs sell)
        side_prob = torch.sigmoid(side).detach().numpy()  # Apply sigmoid
        side_label = 'buy' if side_prob[0] > side_prob[1] else 'sell'
        
        # Generate trade data
        trade = {
            'timestamp': orderbook['lastUpdated'],
            'id': idx + 1,  # Generate trade ID (arbitrary here, can use random or sequential)
            'price': price.item(),
            'side': side_label,
            'amount': amount.item()
        }
        
        trades.append(trade)
    
    # Convert trades list to pandas DataFrame
    trades_df = pd.DataFrame(trades)
    return trades_df
