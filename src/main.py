import pandas as pd

from helpers.data_loader import read_data
from classes.market_maker import MarketMaker
from helpers.charts import *

# Set pandas display option to show 8 digits after the decimal point
pd.set_option('display.float_format', '{:.8f}'.format)

print("Loading orderbook and trading data ...")
orderbook_df, trades_df = read_data()
print("Loading completed ...\n")

print("Initializing the market maker ...")
market_maker = MarketMaker(orderbook_df, trades_df, fair_price_method='wmid', spread_pct=50)

# Compute all fair values and quotes at once
market_maker.compute_all_quotes()

# Retrieve and print the fair prices and quotes DataFrame
all_quotes_df = market_maker.fair_prices_df
candlestick_chart(trades_df=trades_df, market_maker=market_maker, time_interval='1D')
spread_chart(orderbook_df=orderbook_df, market_maker=market_maker, time_interval='1D')
prices_chart(trades_df=trades_df, market_maker=market_maker, time_interval='1min')
plot_volume_analysis(trades_df=trades_df)
plot_orderbook_depth(orderbook_df=orderbook_df)