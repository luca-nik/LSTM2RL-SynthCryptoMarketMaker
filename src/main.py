import sys
import pandas as pd
import torch

from helpers.data_loader import read_data
from helpers.data_handler import prepare_data
from helpers.configurator import load_config

from classes.orderbook_generator import  orderbook_model_load_or_train

def main():
    # Load config file
    CONFIG = load_config()
    
    # Define retraining flag: if True retrai the model
    retrain_model = True
    
    # Set pandas display option to show 8 digits after the decimal point
    pd.set_option('display.float_format', '{:.8f}'.format)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("Loading orderbook and trading data ...")
    orderbook_df, trades_df = read_data(CONFIG)
    print("Loading completed!\n")
    
    print("Preparing datasets ...")
    orderbook_train, trades_train, orderbook_test, trades_test, ob_scaler, tr_scaler = prepare_data(orderbook_df, trades_df, CONFIG)
    #print(orderbook_train.tensors[0].shape)
    #print(orderbook_train.tensors[1].shape)
    #print(orderbook_test.tensors[0].shape)
    #print(orderbook_test.tensors[1].shape)
    print("Datasets prepared succesfully!\n")
    
    # Initialize the model
    orderbook_model = orderbook_model_load_or_train(orderbook_train, CONFIG, retrain_model, device, ob_scaler)
    print("Testing the model ...")
    #orderbook_model.test(orderbook_test, device, ob_scaler)

    
    #trades_train_loader = DataLoader(trades_train, batch_size=CONFIG['training']['batch_size'], shuffle=False)

if __name__ == "__main__":
    main()