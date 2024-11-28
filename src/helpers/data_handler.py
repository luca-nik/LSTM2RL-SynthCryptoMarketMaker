from typing import Tuple

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def filter_relevant_features_and_split_data(
    orderbook_df: pd.DataFrame, 
    trades_df: pd.DataFrame,
    CONFIG: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the orderbook and trades data, extracts relevant features, 
    and splits the data into training and testing sets.

    Args:
        orderbook_df (pd.DataFrame): DataFrame containing the orderbook data with 'asks' and 'bids'.
        trades_df (pd.DataFrame): DataFrame containing the trade data with 'price', 'side', and 'amount'.
        CONFIG (dict): configuration dictionary

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - Training orderbook data (pd.DataFrame).
            - Training trades data (pd.DataFrame).
            - Testing orderbook data (pd.DataFrame).
            - Testing trades data (pd.DataFrame).
    """

    PCT_TRAIN = CONFIG['data']['pct_train']

    # Preprocess the orderbook data to extract the best bid and ask information
    preprocess_orderbook_df = orderbook_df.apply(orderbook_extract_bid_ask, axis=1)
    
    # Preprocess the trades data to extract price, side, and amount information
    preprocess_trades_df = trades_df.apply(trades_extract_price_side_amount, axis=1)

    # Rename columns for clarity
    preprocess_orderbook_df.columns = ['Best Ask', 'Best Ask Volume', 'Best Bid', 'Best Bid Volume']
    preprocess_trades_df.columns = ['Price', 'Amount', 'Side']

    # Calculate the train-test split index
    train_size = int(PCT_TRAIN * len(orderbook_df))

    # Split the preprocessed data into training and testing sets
    orderbook_train = preprocess_orderbook_df[:train_size]
    trades_train = preprocess_trades_df[:train_size]
    orderbook_test = preprocess_orderbook_df[train_size:]
    trades_test = preprocess_trades_df[train_size:]

    return orderbook_train, trades_train, orderbook_test, trades_test

def orderbook_extract_bid_ask(row: pd.Series) -> pd.Series:
    """
    Extracts the best ask, best ask volume, best bid, and best bid volume 
    from a single row of the orderbook DataFrame.

    Args:
        row (pd.Series): A row from the orderbook DataFrame containing 'asks' and 'bids'.

    Returns:
        pd.Series: A Series with the best ask, best ask volume, best bid, and best bid volume.
    """
    best_ask = row['asks'][0][0]  # First ask price
    best_ask_volume = row['asks'][0][1]  # First ask volume
    best_bid = row['bids'][0][0]  # First bid price
    best_bid_volume = row['bids'][0][1]  # First bid volume
    return pd.Series([best_ask, best_ask_volume, best_bid, best_bid_volume])

def trades_extract_price_side_amount(row: pd.Series) -> pd.Series:
    """
    Extracts the price, side (buy/sell), and amount from a single row of the trades DataFrame.

    Args:
        row (pd.Series): A row from the trades DataFrame containing 'price', 'side', and 'amount'.

    Returns:
        pd.Series: A Series with price, amount, and side (1 for sell, 0 for buy).
    """
    price = row['price']
    side = 1 if row['side'] == 'sell' else 0  # Side is encoded as 1 for sell, 0 for buy
    amount = row['amount']
    return pd.Series([price, amount, side])

def create_sequences(orderbook_data: pd.DataFrame, trades_data: pd.DataFrame, CONFIG: dict):
    """
    Create sequences of data (with historical context) for training.
    
    Args:
        orderbook_data (pd.DataFrame): DataFrame containing order book features.
        trades_data (pd.DataFrame): DataFrame containing trade features.
        CONFIG (dict): configuration dictionary
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            Sequences of orderbook, trades data, and target values.
    """
    seq_length = CONFIG['data']['seq_length']
    orderbook_sequences = []
    trades_sequences = []
    target_orderbook_sequences = []
    target_trades_sequences = []

    for i in range(len(orderbook_data) - seq_length):
        orderbook_seq = orderbook_data.iloc[i:i+seq_length].values
        target_orderbook = orderbook_data.iloc[i+seq_length]  # Target is next timestep's best ask and bid
        orderbook_sequences.append(orderbook_seq)
        target_orderbook_sequences.append(target_orderbook)
        
    for i in range(len(trades_data) - seq_length):
        trades_seq = trades_data.iloc[i:i+seq_length].values
        target_trades = trades_data.iloc[i+seq_length]  # Target is next timestep's variables
        trades_sequences.append(trades_seq)
        target_trades_sequences.append(target_trades)

    # Convert lists to numpy arrays before converting to tensors
    orderbook_sequences = np.array(orderbook_sequences)
    trades_sequences = np.array(trades_sequences)
    target_orderbook_sequences = np.array(target_orderbook_sequences)
    target_trades_sequences = np.array(target_trades_sequences)

    return torch.tensor(orderbook_sequences, dtype=torch.float32), \
           torch.tensor(trades_sequences, dtype=torch.float32), \
           torch.tensor(target_orderbook_sequences, dtype=torch.float32), \
           torch.tensor(target_trades_sequences, dtype=torch.float32)

# dcpr
def standardize_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardizes the data using Z-score standardization.
    
    Args:
        train_data (pd.DataFrame): Training data to fit the scaler.
        test_data (pd.DataFrame): Test data to apply the scaler.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            Standardized train and test data.
    """
    scaler = StandardScaler()
    
    # Fit the scaler on training data
    train_data_scaled = scaler.fit_transform(train_data)
    
    # Apply the same transformation to the test data
    test_data_scaled = scaler.transform(test_data)
    
    # Convert back to DataFrame for ease of use
    train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data.columns)
    test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.columns)
    
    return train_data_scaled, test_data_scaled

def prepare_data(orderbook_df: pd.DataFrame, trades_df: pd.DataFrame, CONFIG: dict):


    # Preprocess the data and split them into train test
    orderbook_train, trades_train, orderbook_test, trades_test = filter_relevant_features_and_split_data(orderbook_df, trades_df, CONFIG)
    
    # Step 1: Standardize the Data
    orderbook_train_scaled, orderbook_test_scaled = standardize_data(orderbook_train, orderbook_test)
    trades_train_scaled, trades_test_scaled = standardize_data(trades_train, trades_test)
    
    # Step 2: Create Sequences for training 
    orderbook_train_sequences, trades_train_sequences, orderbook_train_target_sequences, trades_train_target_sequences = \
        create_sequences(orderbook_train_scaled, trades_train_scaled, CONFIG)
    # Sequences for test
    orderbook_test_sequences, trades_test_sequences, orderbook_test_target_sequences, trades_test_target_sequences = \
        create_sequences(orderbook_test_scaled, trades_test_scaled, CONFIG)
    
    # Step 3: Prepare DataLoader for Training
    orderbook_train_dataset = TensorDataset(orderbook_train_sequences, orderbook_train_target_sequences)
    trades_train_dataset = TensorDataset(trades_train_sequences, trades_train_target_sequences)
    # And for test
    orderbook_test_dataset = TensorDataset(orderbook_test_sequences, orderbook_test_target_sequences)
    trades_test_dataset = TensorDataset(trades_test_sequences, trades_test_target_sequences)

    return orderbook_train_dataset, trades_train_dataset, orderbook_test_dataset, trades_test_dataset
