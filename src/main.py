import sys
import pandas as pd
import torch

from data.loader import read_data
from data.processer import prepare_data
from utils.configurator import load_config

from models.orderbook_generator import  orderbook_model_load_or_train
from models.trades_generator import  trades_model_load_or_train

def main():
    # Load config file
    CONFIG = load_config()
    
    # Define retraining flag: if True retrai the model
    train_orderbook = False
    train_trades = False
    
    # Set pandas display option to show 8 digits after the decimal point
    pd.set_option('display.float_format', '{:.8f}'.format)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("Loading orderbook and trading data ...")
    orderbook_df, trades_df = read_data(CONFIG)
    print("Loading completed!\n")
    
    print("Preparing datasets ...")
    orderbook_train, trades_train, orderbook_test, trades_test, orderbook_scaler, trades_scaler = prepare_data(orderbook_df, trades_df, CONFIG)
    #print(trades_train.tensors[0].shape)
    #print(trades_train.tensors[1].shape)
    #print(trades_test.tensors[0].shape)
    #print(trades_test.tensors[1].shape)
    #sys.exit()
    print("Datasets prepared succesfully!\n")
    
    # Load trained or train the models
    print("Loading or training the models ...")
    orderbook_model = orderbook_model_load_or_train(orderbook_train, CONFIG, train_orderbook, device, orderbook_scaler)
    trades_model = trades_model_load_or_train(trades_train, CONFIG, train_trades, device, trades_scaler)
    
    # Test the models
    print("Testing the models ...")
    orderbook_model.test(orderbook_test, device)
    trades_model.test(trades_test, device)


    
    #trades_train_loader = DataLoader(trades_train, batch_size=CONFIG['training']['batch_size'], shuffle=False)

if __name__ == "__main__":
    main()