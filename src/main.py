import sys
import pandas as pd

from constants import PCT_TRAIN

from helpers.data_loader import read_data  # Assuming you have a read_data function in your helpers module
from helpers.data_handler import preprocess_and_split_data, create_sequences, standardize_data

import matplotlib.pyplot as plt


# Set pandas display option to show 8 digits after the decimal point
pd.set_option('display.float_format', '{:.8f}'.format)

# Print a message indicating that data loading is starting
print("Loading orderbook and trading data ...")

# Load the orderbook and trading data
orderbook_df, trades_df = read_data()

# Print a message indicating that the loading process is completed
print("Loading completed ...\n")

# Preprocess the data and split them into train test
orderbook_train, trades_train, orderbook_test, trades_test = preprocess_and_split_data(orderbook_df, trades_df, PCT_TRAIN)

# Step 1: Create Sequences
seq_length = 10  # Using the last 10 timesteps as a sequence
orderbook_sequences, trades_sequences, target_sequences = create_sequences(orderbook_train, trades_train, seq_length)

# Step 2: Standardize the Data
orderbook_train_scaled, orderbook_test_scaled = standardize_data(orderbook_train, orderbook_test)
trades_train_scaled, trades_test_scaled = standardize_data(trades_train, trades_test)

# Step 3: Convert to Sequences after Scaling
orderbook_sequences_scaled, trades_sequences_scaled, target_sequences_scaled = create_sequences(
    orderbook_train_scaled, trades_train_scaled, seq_length)

## Step 4: Prepare DataLoader for Training
#train_dataset = TensorDataset(orderbook_sequences_scaled, trades_sequences_scaled, target_sequences_scaled)
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Now you're ready to use `train_loader` in your training loop

#plt.plot(orderbook_train['Best Ask'], label='Best Ask', color = 'r', linewidth = 0.5)
#plt.plot(orderbook_train['Best Ask'], label='Best Bid', color = 'b', linewidth = 0.5)
#plt.legend()
#plt.savefig('images/example.png', dpi=400) 
### Train orderbook


#
## Now, `orderbook_df` contains only the 'lastUpdated', 'best_ask', and 'best_bid' columns
#orderbook_df = preprocess_orderbook_data(orderbook_df)
#
## Split the data into train and test sets (3/4 for training, 1/4 for testing)
#train_size = int(0.75 * len(orderbook_df))  # 3/4 of the data
#orderbook_train = orderbook_df[:train_size]
#orderbook_test = orderbook_df[train_size:]
#
#trades_train = trades_df[:train_size]
#trades_test = trades_df[train_size:]
#
## Initialize and train the OrderBook Generator model
#orderbook_input_dim = len(orderbook_df.columns)*2  
#orderbook_hidden_dim = 64  # You can adjust this hyperparameter
#orderbook_model = OrderBookGenerator(orderbook_input_dim, orderbook_hidden_dim)
##
#print("Training OrderBook Generator...")
#train_order_book_generator(orderbook_model, orderbook_train)
##
## Generate orderbook predictions for the training set and test set
#print("Generating OrderBook predictions...")
#
# Training set orderbooks
#train_orderbook_predictions = generate_orderbook(orderbook_model, orderbook_train, num_steps=len(orderbook_train))
#
## Test set orderbooks
#test_orderbook_predictions = generate_orderbook(orderbook_model, orderbook_test, num_steps=len(orderbook_test))

## Initialize and train the Trades Generator model
#trades_input_dim = 5  # Adjust based on the number of features (e.g., best bid, best ask, etc.)
#trades_hidden_dim = 64  # You can adjust this hyperparameter
#trades_model = TradesGenerator(trades_input_dim, trades_hidden_dim)
#
#print("Training Trades Generator...")
#train_trades_generator(trades_model, trades_train.values.tolist(), epochs=100)
#
## Generate trade predictions for the training set and test set
#print("Generating Trade predictions...")
#
## Training set trades
#train_trades_predictions = generate_trades(trades_model, orderbook_train, num_trades=len(trades_train))
#
## Test set trades
#test_trades_predictions = generate_trades(trades_model, orderbook_test, num_trades=len(trades_test))
#
## Plotting Results
#
## Make sure the directory exists for saving images
#os.makedirs('images/ML', exist_ok=True)
#
## Plot OrderBook predictions for training vs actual
#plt.figure(figsize=(12, 6))
#
#plt.subplot(1, 2, 1)
#plt.plot(orderbook_train['lastUpdated'], orderbook_train['bids'].apply(lambda x: x[0][0]), label="Actual Best Bid")
#plt.plot(orderbook_train['lastUpdated'], [order['bids'][0][0] for order in train_orderbook_predictions], label="Predicted Best Bid")
#plt.title("OrderBook - Training Set")
#plt.xlabel("Timestamp")
#plt.ylabel("Best Bid Price")
#plt.legend()
#
## Plot OrderBook predictions for test vs actual
#plt.subplot(1, 2, 2)
#plt.plot(orderbook_test['lastUpdated'], orderbook_test['bids'].apply(lambda x: x[0][0]), label="Actual Best Bid")
#plt.plot(orderbook_test['lastUpdated'], [order['bids'][0][0] for order in test_orderbook_predictions], label="Predicted Best Bid")
#plt.title("OrderBook - Test Set")
#plt.xlabel("Timestamp")
#plt.ylabel("Best Bid Price")
#plt.legend()
#
#plt.tight_layout()
#plt.savefig('images/ML/orderbook_predictions.png')  # Save the plot
#plt.show()
#
## Plot Trades predictions for training vs actual
#plt.figure(figsize=(12, 6))
#
#plt.subplot(1, 2, 1)
#plt.plot(trades_train['timestamp'], trades_train['price'], label="Actual Trade Price")
#plt.plot(trades_train['timestamp'], train_trades_predictions['price'], label="Predicted Trade Price")
#plt.title("Trades - Training Set")
#plt.xlabel("Timestamp")
#plt.ylabel("Trade Price")
#plt.legend()
#
## Plot Trades predictions for test vs actual
#plt.subplot(1, 2, 2)
#plt.plot(trades_test['timestamp'], trades_test['price'], label="Actual Trade Price")
#plt.plot(trades_test['timestamp'], test_trades_predictions['price'], label="Predicted Trade Price")
#plt.title("Trades - Test Set")
#plt.xlabel("Timestamp")
#plt.ylabel("Trade Price")
#plt.legend()
#
#plt.tight_layout()
#plt.savefig('images/ML/trades_predictions.png')  # Save the plot
#plt.show()
#
## Evaluate performance (Mean Squared Error)
#from sklearn.metrics import mean_squared_error
#
## Evaluate OrderBook performance
#train_orderbook_actual = orderbook_train['bids'].apply(lambda x: x[0][0]).values
#train_orderbook_predicted = [order['bids'][0][0] for order in train_orderbook_predictions]
#test_orderbook_actual = orderbook_test['bids'].apply(lambda x: x[0][0]).values
#test_orderbook_predicted = [order['bids'][0][0] for order in test_orderbook_predictions]
#
#train_orderbook_mse = mean_squared_error(train_orderbook_actual, train_orderbook_predicted)
#test_orderbook_mse = mean_squared_error(test_orderbook_actual, test_orderbook_predicted)
#
#print(f"OrderBook Model - Train MSE: {train_orderbook_mse}")
#print(f"OrderBook Model - Test MSE: {test_orderbook_mse}")
#
## Evaluate Trades performance
#train_trades_actual = trades_train['price'].values
#train_trades_predicted = train_trades_predictions['price'].values
#test_trades_actual = trades_test['price'].values
#test_trades_predicted = test_trades_predictions['price'].values
#
#train_trades_mse = mean_squared_error(train_trades_actual, train_trades_predicted)
#test_trades_mse = mean_squared_error(test_trades_actual, test_trades_predicted)
#
#print(f"Trades Model - Train MSE: {train_trades_mse}")
#print(f"Trades Model - Test MSE: {test_trades_mse}")
#