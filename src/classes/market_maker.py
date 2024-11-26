import pandas as pd
from typing import Optional

class MarketMaker:
    def __init__(self, orderbook_df: pd.DataFrame, trade_data: Optional[pd.DataFrame] = None,
                 fair_price_method: str = 'mid', spread_pct: float = 1):
        """
        Initializes the MarketMaker object.
        
        Args:
            orderbook_df (pd.DataFrame): Order book DataFrame containing bids and asks.
            trade_data (Optional[pd.DataFrame]): Trade DataFrame containing past trades (default is None).
            fair_price_method (str): Method for calculating the fair value. Options are 'mid' (midpoint),
                                      'vwap' (volume-weighted average price), etc.
            spread_pct (float): The spread around the fair value as a percentage of the computed fair value.
        """
        self.orderbook_df = orderbook_df
        self.trade_data = trade_data if trade_data is not None else pd.DataFrame()
        self.fair_price_method = fair_price_method
        self.spread_pct = spread_pct

        # Store the calculated fair values and quotes in a DataFrame for easy access
        self.fair_prices_df = pd.DataFrame(columns=['timestamp', 'fair_value', 'bid_price', 'ask_price'])

    def calculate_fair_value(self, current_orderbook: pd.Series) -> float:
        """
        Calculate the fair value based on the selected method.
        
        Args:
            current_orderbook (pd.Series): A row from the orderbook containing bid and ask data.
        
        Returns:
            float: The predicted fair value.
        """
        if self.fair_price_method == 'mid':
            # Simple fair price calculation: midpoint between best bid and best ask
            best_bid = current_orderbook['bids'][0][0]  
            best_ask = current_orderbook['asks'][0][0]  
            fair_value = (best_bid + best_ask) / 2
        elif self.fair_price_method == 'wmid':
            # Weighted mid price (wmid)
            best_bid_price, best_bid_quantity = current_orderbook['bids'][0] 
            best_ask_price, best_ask_quantity = current_orderbook['asks'][0]  
            fair_value = (
                (best_bid_price * best_bid_quantity) + (best_ask_price * best_ask_quantity)
            ) / (best_bid_quantity + best_ask_quantity) if (best_bid_quantity + best_ask_quantity) > 0 else 0
        else:
            raise ValueError(f"Unknown fair price method: {self.fair_price_method}")
        return fair_value

    def generate_quotes(self, fair_value: float, current_orderbook: pd.Series) -> dict:
        """
        Generate quotes for the market maker (bid and ask prices) around the fair value.
        
        Args:
            fair_value (float): The predicted fair value.
            current_orderbook (pd.Series): A row from the orderbook containing bid and ask data.
        
        Returns:
            dict: A dictionary with bid and ask quotes.
        """
        # Extract the best bid and ask prices from the orderbook
        best_bid_price = current_orderbook['bids'][0][0]  # The best bid price
        best_ask_price = current_orderbook['asks'][0][0]  # The best ask price
        # Calculate the spreads
        bid_spread = fair_value - best_bid_price  
        ask_spread = best_ask_price - fair_value  
        # Calculate quotes
        bid_price = fair_value  - (self.spread_pct/100) * bid_spread
        ask_price = fair_value  + (self.spread_pct/100) * ask_spread
        return {'bid': bid_price, 'ask': ask_price}

    def compute_all_quotes(self):
        """
        Computes all fair prices and orders for the entire orderbook dataset.
        The results are stored in a DataFrame inside the market maker object.
        
        Returns:
            pd.DataFrame: DataFrame containing fair prices, bids, and asks for each time step.
        """
        all_quotes = []

        # Iterate through all rows in the orderbook DataFrame getting the orderbook at time t
        for idx, orderbook_t in self.orderbook_df.iterrows():
            fair_value = self.calculate_fair_value(orderbook_t)  # Calculate fair price for current time
            quotes = self.generate_quotes(fair_value, orderbook_t)# Generate bid/ask quotes
            
            # Store the timestamp, fair price, and the generated bid/ask
            all_quotes.append({
                'timestamp': orderbook_t['lastUpdated'],
                'fair_value': fair_value,
                'bid_price': quotes['bid'],
                'ask_price': quotes['ask'],
            })
        
        # Convert the list of quotes into a DataFrame
        self.fair_prices_df = pd.DataFrame(all_quotes)