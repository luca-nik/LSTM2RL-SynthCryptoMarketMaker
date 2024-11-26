import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

from classes.market_maker import MarketMaker


def candlestick_chart(trades_df: pd.DataFrame, market_maker: MarketMaker, time_interval: str = '60min') -> None:
    """
    Creates and saves a candlestick chart from trade data and overlays the market maker's bid and ask prices.

    Args:
        trades_df (pd.DataFrame): DataFrame containing trade data.
        market_maker (MarketMaker): The MarketMaker object to access the quotes.
        time_interval (str): Time interval for resampling, e.g., '60min'.
    """
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    ohlc = trades_df.set_index('timestamp')['price'].resample(time_interval).ohlc()
    
    # Extract market maker quotes (assuming 'all_quotes_df' is available in MarketMaker)
    quotes_df = market_maker.fair_prices_df
    quotes_df['timestamp'] = pd.to_datetime(quotes_df['timestamp'])
    quotes_df = quotes_df.set_index('timestamp').resample(time_interval).last()

    # Define bid and ask as additional lines to overlay
    bid_line = mpf.make_addplot(quotes_df['bid_price'], color='blue', linestyle='--', width=1, label = "MM's bid quote")
    ask_line = mpf.make_addplot(quotes_df['ask_price'], color='red', linestyle='--', width=1, label = "MM's ask quote")

    # Plot candlestick chart without title and with tight layout
    mpf.plot(ohlc, type='candle', style='charles', ylabel='Price', ylabel_lower='Volume', 
             addplot=[bid_line, ask_line], figsize=(10, 6), 
             savefig='images/candlestick_with_quotes.png', 
             tight_layout=True, 
             title="",  
             )

    # Add legend manually after plotting
    
    #print("Candlestick chart with market maker quotes saved to images/candlestick_with_quotes.png.")

def spread_chart(orderbook_df: pd.DataFrame, market_maker: MarketMaker, time_interval: str = '60min') -> None:
    """
    Creates and saves a bid-ask spread chart and overlays the market maker's bid-ask spread.
    
    Args:
        orderbook_df (pd.DataFrame): DataFrame containing order book data.
        market_maker (MarketMaker): The MarketMaker object to access the quotes.
        time_interval (str): Time interval for resampling.
    """
    # Process and resample orderbook data
    orderbook_df['lastUpdated'] = pd.to_datetime(orderbook_df['lastUpdated'])
    orderbook_df['best_bid'] = orderbook_df['bids'].apply(lambda x: x[0][0])
    orderbook_df['best_ask'] = orderbook_df['asks'].apply(lambda x: x[0][0])
    orderbook_df['spread'] = orderbook_df['best_ask'] - orderbook_df['best_bid']
    
    # Resample to align with the specified time interval
    spread_resampled = orderbook_df.set_index('lastUpdated')['spread'].resample(time_interval).mean()

    # Process and resample market maker quotes
    quotes_df = market_maker.fair_prices_df
    quotes_df['timestamp'] = pd.to_datetime(quotes_df['timestamp'])
    quotes_df = quotes_df.set_index('timestamp').resample(time_interval).last()

    # Calculate market maker's spread based on bid and ask prices
    quotes_df['market_maker_spread'] = quotes_df['ask_price'] - quotes_df['bid_price']

    # Reindex both DataFrames to ensure alignment on the same time intervals
    aligned_spread = spread_resampled.reindex(quotes_df.index, method='pad')
    aligned_quotes_spread = quotes_df['market_maker_spread'].reindex(aligned_spread.index, method='pad')

    # Plot the order book spread and market maker's spread
    plt.figure(figsize=(10, 5))
    plt.plot(aligned_spread.index, aligned_spread, label="Orderbook Bid-Ask Spread", color='purple')
    plt.plot(aligned_quotes_spread.index, aligned_quotes_spread, label="Market Maker Bid-Ask Spread", color='orange', linestyle='--')
    
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.title('Bid-Ask Spread Comparison (Market Maker vs Orderbook)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig("images/bid_ask_spread_with_market_maker.png")
    #print("Bid-Ask spread chart with market maker quotes saved to images/bid_ask_spread_with_market_maker.png.")


def prices_chart(trades_df: pd.DataFrame, market_maker: MarketMaker, time_interval: str = '60min') -> None:
    """
    Plots the trend of the fair price from the order book with respect to the trade price over time.
    Calculates and displays the percentage difference between the trade and fair prices in a subplot below the main plot.

    Args:
        trades_df (pd.DataFrame): DataFrame containing trade data.
        market_maker (MarketMaker): The MarketMaker object to access the quotes.
        time_interval (str): Time interval for resampling the data (e.g., '60min').
    """
    # Ensure timestamps are in datetime format
    market_maker.fair_prices_df['timestamp'] = pd.to_datetime(market_maker.fair_prices_df['timestamp'])
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    # Resample both dataframes to the specified time interval
    fair_price_resampled = market_maker.fair_prices_df.set_index('timestamp')['fair_value'].resample(time_interval).last()
    trade_price_resampled = trades_df.set_index('timestamp')['price'].resample(time_interval).last()
    
    # Align the two series to calculate the percentage difference
    aligned_trade_price = trade_price_resampled.reindex(fair_price_resampled.index, method='pad')
    aligned_fair_price = fair_price_resampled.reindex(trade_price_resampled.index, method='pad')
    aligned_trade_price = aligned_trade_price.dropna()
    aligned_fair_price = aligned_fair_price.dropna()
    
    # Calculate the percentage difference between trade and fair prices
    percentage_difference = (aligned_trade_price - aligned_fair_price) / aligned_trade_price * 100
    
    # Create a figure with two subplots (one for prices, one for percentage difference)
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot fair price from orderbook
    ax1.plot(fair_price_resampled.index, fair_price_resampled, label="Fair Price", color='b', marker='o', markersize=2, linewidth=0.5)
    
    # Plot trade price
    ax1.plot(trade_price_resampled.index, trade_price_resampled, label="Trade Price", color='r', marker='o', markersize=1, linewidth=0.2, alpha = 0.5)
    
    # Customize first subplot (Price trend plot)
    ax1.set_title(f"Fair Price vs Trade Price (Interval: {time_interval})")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    
    # Plot percentage difference in the second subplot (below)
    print('Maximum price difference: {:.3f} %'.format(max(abs(percentage_difference))))
    ax3.plot(percentage_difference.index, percentage_difference, label="Percentage Difference", color='g', linestyle='--', linewidth=1)
    ax3.set_title("Percentage Difference Between Trade and Fair Prices")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Percentage Difference (%)")
    ax3.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("images/fair_price_vs_trade_price.png", dpi=400)
    #print("Plot saved to images/fair_price_vs_trade_price_with_percentage_difference_below.png")

def plot_volume_analysis(trades_df: pd.DataFrame) -> None:
    """
    Creates a chart showing trading volume by side (buy/sell) on a daily basis.
    """
    plt.figure(figsize=(12, 6))
    
    # Convert timestamp to datetime and get daily dates
    trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
    
    # Group trades by date and side, sum the volumes
    volume_by_side = trades_df.groupby(['date', 'side'], observed=False)['amount'].sum().unstack()

    
    # Create stacked bar chart
    ax = volume_by_side.plot(
        kind='bar', 
        stacked=True,
        color=['green', 'red']  # green for buy, red for sell
    )
    
    # Customize the plot
    plt.title('Daily Trading Volume by Side')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    plt.legend(title='Trade Side')
    
    # Add volume numbers on top of each bar
    total_volumes = volume_by_side.sum(axis=1)
    for i, total in enumerate(total_volumes):
        plt.text(i, total, f'{total:,.0f}', 
                ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/daily_volume_analysis.png')
    
    # Print buy sell ratio
    print("\nBuy/Sell Ratio by Day:")
    buy_sell_ratio = volume_by_side['buy'] / volume_by_side['sell']
    for date, ratio in buy_sell_ratio.items():
        print(f"{date}: {ratio}")

def plot_orderbook_depth(orderbook_df: pd.DataFrame) -> None:
    """
    Plots daily average bid and ask depth from the order book.
    """
    def calculate_daily_depth(row, levels=10):
        bid_depth = sum(bid[1] for bid in row['bids'][:levels])
        ask_depth = sum(ask[1] for ask in row['asks'][:levels])
        return pd.Series({'bid_depth': bid_depth, 'ask_depth': ask_depth})
    
    # Apply the depth calculation for each row
    depth_df = orderbook_df.apply(calculate_daily_depth, axis=1)
    depth_df['date'] = pd.to_datetime(orderbook_df['lastUpdated']).dt.date
    
    # Calculate daily averages
    daily_depth = depth_df.groupby('date').mean()
    
    # Plot daily depth over time
    plt.figure(figsize=(12, 6))
    plt.plot(daily_depth.index, daily_depth['bid_depth'], label='Bid Depth', color='g')
    plt.plot(daily_depth.index, daily_depth['ask_depth'], label='Ask Depth', color='r')
    plt.title('Daily Order Book Depth')
    plt.xlabel('Date')
    plt.ylabel('Average Cumulative Volume')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/daily_orderbook_depth.png')
