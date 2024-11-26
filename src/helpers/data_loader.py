from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_env_variable(var_name: str, default: Union[str, None] = None) -> str:
    """
    Helper function to fetch environment variable or return a default value.
    
    Args:
        var_name (str): Name of the environment variable.
        default (str, optional): Default value if the variable is not found. Defaults to None.
    
    Returns:
        str: Value of the environment variable.
    """
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Missing environment variable: {var_name}")
    return value

# Fetch paths using the helper function
trades_file_path = get_env_variable("TRADES_FILE_PATH", "attachments/trades.csv")
trades_save_path = get_env_variable("TRADES_SAVE_PATH", "data/trades_df.parquet")
orderbook_files_path = get_env_variable("ORDERBOOK_FILES_PATH", "attachments/binance_iotabtc_orderbooks/")
orderbook_save_path = get_env_variable("ORDERBOOK_SAVE_PATH", "data/full_orderbook_df.parquet")

def read_trades_csv(filepath: Union[str, Path], save_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a CSV file containing trade data and saves it for faster future access.
    
    Args:
        filepath (Union[str, Path]): Path to the trades CSV file.
        save_path (Union[str, Path]): Path where the processed trades DataFrame will be saved.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'id', 'price', 'side', 'amount']
    """
    save_path = Path(save_path)

    # Check if the DataFrame has already been saved
    if save_path.exists():
        print(f"Loading trades DataFrame from {save_path}")
        if save_path.suffix == ".parquet":
            return pd.read_parquet(save_path)
        elif save_path.suffix == ".pkl":
            return pd.read_pickle(save_path)
        else:
            raise ValueError("Unsupported file format. Use either .parquet or .pkl")

    # Otherwise, read the trades CSV file and save it
    print(f"Processing and saving trades DataFrame to {save_path}")
    trades_df = pd.read_csv(filepath, index_col=0)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df['side'] = trades_df['side'].astype('category')

    # Save DataFrame to disk
    if save_path.suffix == ".parquet":
        trades_df.to_parquet(save_path, index=False)
    elif save_path.suffix == ".pkl":
        trades_df.to_pickle(save_path)
    else:
        raise ValueError("Unsupported file format. Use either .parquet or .pkl")

    return trades_df

def read_orderbook_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a CSV file containing orderbook data with nested lists in 'asks' and 'bids' columns.
    
    Args:
        filepath (Union[str, Path]): Path to the orderbook CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['lastUpdated', 'asks', 'bids'],
                      where 'asks' and 'bids' are lists of tuples (price, quantity).
    """
    def parse_nested_list(column_value: str) -> List[Tuple[float, int]]:
        """Helper function to parse a nested list from a string format in the CSV."""
        return eval(column_value)

    orderbook_df = pd.read_csv(filepath)
    orderbook_df['lastUpdated'] = pd.to_datetime(orderbook_df['lastUpdated'])
    orderbook_df['asks'] = orderbook_df['asks'].apply(parse_nested_list)
    orderbook_df['bids'] = orderbook_df['bids'].apply(parse_nested_list)
    return orderbook_df

def concatenate_orderbook_series(folder_path: Union[str, Path], save_path: Union[str, Path]) -> pd.DataFrame:
    """
    Concatenates multiple orderbook CSV files from a specified folder into a single DataFrame and saves it.
    
    Args:
        folder_path (Union[str, Path]): Path to the folder containing the orderbook CSV files.
        save_path (Union[str, Path]): Path where the concatenated DataFrame will be saved.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame with data from all files in the folder.
    """
    save_path = Path(save_path)

    # Check if the DataFrame has already been saved
    if save_path.exists():
        print(f"Loading concatenated orderbook DataFrame from {save_path}")
        if save_path.suffix == ".parquet":
            return pd.read_parquet(save_path)
        elif save_path.suffix == ".pkl":
            return pd.read_pickle(save_path)
        else:
            raise ValueError("Unsupported file format. Use either .parquet or .pkl")

    # Otherwise, read, concatenate, and save
    print(f"Processing and saving concatenated orderbook DataFrame to {save_path}")
    folder_path = Path(folder_path)
    all_files = sorted(folder_path.glob("*.csv")) 

    # Use list comprehension to read each file and store in a list
    orderbook_dfs = [read_orderbook_csv(file) for file in all_files]

    # Concatenate all DataFrames into a single DataFrame
    concatenated_df = pd.concat(orderbook_dfs, ignore_index=True)

    # Save DataFrame to disk
    if save_path.suffix == ".parquet":
        concatenated_df.to_parquet(save_path, index=False)
    elif save_path.suffix == ".pkl":
        concatenated_df.to_pickle(save_path)
    else:
        raise ValueError("Unsupported file format. Use either .parquet or .pkl")

    return concatenated_df

def read_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the trades and order book data using paths stored in environment variables.
    
    Returns:
        pd.DataFrame: DataFrame containing the order book data.
        pd.DataFrame: DataFrame containing the trade data.
    """
    
    # Read trades and orderbook data, save them for quicker processing
    trades_df = read_trades_csv(trades_file_path, trades_save_path)
    orderbook_df = concatenate_orderbook_series(orderbook_files_path, orderbook_save_path)

    # Ensure both dataframes are sorted by timestamps
    orderbook_df = orderbook_df.sort_values('lastUpdated')
    trades_df = trades_df.sort_values('timestamp')

    return orderbook_df, trades_df
