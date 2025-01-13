Here's the updated and detailed **README** that incorporates all the requested elements:

---

![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)  
# LSTM2RL-SynthCryptoMarketMaker  

## Overview  
**`LSTM2RL-SynthCryptoMarketMaker`** is a personal project combining deep learning and reinforcement learning to create a synthetic crypto market for research and experimentation.  

### Key Components:  
1. **LSTMs**: Two neural networks trained on real market data from [Binance IOTA/BTC Spot Market](https://www.binance.com/en/trade/IOTA_BTC?type=spot).  
   - **OrderBookGenerator**: Simulates order book dynamics.  
   - **TradesGenerator**: Simulates trade execution within the order book.  
2. **Surrogate Market**: A synthetic environment where the models act as oracles to mimic the Binance IOTA/BTC market.  
3. **RL Market Maker**: An RL agent that learns to optimize its market-making strategy by interacting with the surrogate market.

This approach provides a controlled, data-driven framework for testing market-making strategies in a safe simulated environment.  

---

## Key Features  
- **Deep Learning Models**: LSTMs replicate order book and trade flow dynamics using real data.  
- **Surrogate Oracles**: The trained models simulate market behavior and allow for reinforcement learning experiments.  
- **Reinforcement Learning**: Develop and train an RL agent to adaptively quote in the simulated market.  
- **Flexible Experimentation**: Modify hyperparameters, retrain models, and test new strategies seamlessly.  

---

## Usage  

### Prerequisites  
- **Python** (3.8 or higher recommended)  
- **Poetry** for dependency management  

### Getting Started  
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/luca-nik/LSTM2RL-SynthCryptoMarketMaker.git
   cd LSTM2RL-SynthCryptoMarketMaker
   ```

2. **Install dependencies**:  
   ```bash
   poetry install
   ```

3. **Run the project**:  
   ```bash
   poetry run python src/main.py
   ```

### Example Output  
Running the above command will:  
- Load trade and order book data from `data/trades_df.parquet` and `data/full_orderbook_df.parquet`.  
- Load pretrained LSTM models from `data/models/orderbook_generator.pth` and `data/models/trades_generator.pth`.  
- Preprocess datasets, test the models, and save performance figures to the `images` folder.  

Sample output:  
```plaintext
Using device: cuda 

Loading orderbook and trading data ...
Loading trades DataFrame from data/trades_df.parquet
Loading concatenated orderbook DataFrame from data/full_orderbook_df.parquet
Loading completed!

Preparing datasets ...
Datasets prepared successfully!

Loading or training the models ...
  Loading pretrained Orderbook model ...
  Orderbook model loaded!

  Loading pretrained Trades model ...
  Trades model loaded!

Testing the models ...
  Testing the Orderbook model
  RMSE: 3822.949
  MAE: 1930.319
  Figures saved successfully!

  Testing the Trades model
  RMSE: 1855.032
  MAE: 511.329
  Figures saved successfully!
```

---

## Advanced Configuration  

### Retraining the Models  
To retrain the models:  
1. Set the flags to `True` in `src/main.py`:  
   ```python
   train_orderbook = True
   train_trades = True
   ```
2. Modify the `config.json` file to adjust hyperparameters:  
   ```json
   {
     "data": {
       "seq_length": 15,
       "pct_train": 0.8
     },
     "model": {
       "hidden_size": 64,
       "num_layers": 1
     },
     "training": {
       "batch_size": 32,
       "epochs": 100,
       "learning_rate": 0.001,
       "early_stopping_patience": 10
     },
     "paths": {
       "orderbook_model_path": "data/models/orderbook_generator.pth",
       "trades_model_path": "data/models/trades_generator.pth",
       "images_path": "images/",
       "trades_files_path": "attachments/trades.csv",
       "trades_save_path": "data/trades_df.parquet",
       "orderbook_files_path": "attachments/binance_iotabtc_orderbooks/",
       "orderbook_save_path": "data/full_orderbook_df.parquet"
     }
   }
   ```

---

## Contributions  
We welcome contributions to enhance this project!  

### Current Needs  
1. **Use Trained Oracles**: Generate a fully functional surrogate Binance market.  
2. **Develop RL Market Maker**: Create the reinforcement learning model for the market maker.  
3. **Train the RL Agent**: Optimize the agent’s quoting strategy using the surrogate market.  

### How to Contribute  
1. Fork the repository.  
2. Create a new branch for your feature or fix (`git checkout -b feature-name`).  
3. Commit your changes (`git commit -m 'Description of changes'`).  
4. Push to your branch (`git push origin feature-name`).  
5. Open a pull request.  

Check the `CONTRIBUTING.md` for detailed guidelines.  

---

## Data Sources  
- **Order Book and Trade Data**: Historical data from the [Binance IOTA/BTC Spot Market](https://www.binance.com/en/trade/IOTA_BTC?type=spot).  
  - Order book data stored in `data/full_orderbook_df.parquet`.  
  - Trade data stored in `data/trades_df.parquet`.  

---

## License  
This project is licensed under the MIT License. See `LICENSE` for more details.

