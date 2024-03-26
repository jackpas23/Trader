# Importing necessary libraries
import yfinance as yf
import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import MACD
from ta.momentum import RSIIndicator
from backtesting import Backtest, Strategy
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import alpaca_trade_api as tradeapi


api = tradeapi.REST('your alpaca api key', 'your alpaca secret key', base_url='https://paper-api.alpaca.markets')

# List of stocks to analyze
stoinks = ['HD','AAPL','COST','ACN','MCD','TMO','TMUS','PM','MS','VZ','NEE','BX','BMY','SPGY','BIIB','MRNA','DFS','PTON','DXCM','MSCI','CRWD','HLT','IDXX','SQ','ROK','DASH','DDOG','SPOT','BKR','HAL','IDXX','SGEN','LNG','FICO','XYL','SPLK','HEI','ICLR','NET','RMD','ZM','BRO','ANSS','IR','PLTR','GWW','WST','PWR','VNC','ANSS','MLM','IR','MDB','HZNB','ZS','XYL','SPY', 'META', 'VLO', 'VMW', 'GOOG', 'MSFT', 'TSLA', 'T', 'ADNT', 'NFLX', 'EBAY', 'AAPL', 'UPS', 'AMZN', 'ADBE', 'PEP', 'PLD','NVDA','LLY','V','TSM','WMT','JPM','NVO','MA','ORCL','CVX','BABA','BAC','ACN','LIN','ABT','SAP','TMUS','HDB','NKE','DIS','TTE','WFC','UPS','VZ','CAT','INTC','MCD','BDX','CB','ABNB','SLB','CVS','ZTS','VRTX','REGN','LRCX','ETN','ADI','UBER','GILD','MMC','MDLZ','BKNG']






















# opt values 'BKNG': {'TPbuy': 1.01, 'TPsell': 0.97, 'buyadd': 0, 'selladd': 0, 'bk1': 2, 'bd1': 1, 'bk2': 2, 'bd2': 2, 'ubl': 4, 'rise': 1, 'cons': 2, 'bo': 2, 'ol': 2, 'peak': 2, 'rsig': 2, 'rsib': 2, 'macd': 1, 'vol': 2}}

keys_of_interest = ['TPbuy', 'TPsell', 'buyadd', 'selladd', 'bk1', 'bd1', 'bk2', 'bd2', 'ubl', 'rise', 'cons', 'bo', 'ol', 'peak', 'rsig', 'rsib','macd','vol']



# This function fetches 30-day historical data for the given stock 
def makeDF(stoink):
    current = yf.Ticker(stoink) # Get ticker data for the given stock
    hist = current.history(period='7d',interval='1m') # Fetch 30-day historical data
    # Creating a DataFrame from the historical data
    df = hist
    
    df.reset_index(inplace=True) 
    df['50_SMA'] = df['Close'].rolling(window=7).mean()# Reset index for the DataFrame
    df['200_SMA'] = df['Close'].rolling(window=35).mean()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['Volume'] = df['Volume']
    return df
     # Return the DataFrame

# This function checks whether the stock is moving or in consolidation phase
def mvCheck(results):
    # Rolling averages
    avgHigh = results['High'].rolling(window=60).mean()
    avgLow = results['Low'].rolling(window=60).mean()
    
    
    # Shift the averages to get the average of the last 30 days on the next day
    avgHigh = avgHigh.shift(1)
    avgLow = avgLow.shift(1)
    
    # Calculate the UBL and LBL for each day
    UBL = avgHigh * 1.005
    LBL = avgLow * .995
    
    # Create signals based on whether the high or low price breaks the bounds
    results['Bound Limits'] = 'Consolidating'
    results.loc[results['High'] > UBL, 'Bound Limits'] = "Risen above UBL"
    results.loc[results['Low'] < LBL, 'Bound Limits'] = "Dropped below LBL"
    
    

# This function checks whether the stock is about to breakout



def check_crossover(results):
    # Select only the last 30 days of results
    

    # Shift the moving averages by one day
    results['50_SMA_prev'] = results['50_SMA'].shift(1)
    results['200_SMA_prev'] = results['200_SMA'].shift(1)

    # Create signals based on crossover
    results['Signal'] = 0.0   # default to 0.0
    results.loc[(results['50_SMA'] > results['200_SMA']) & (results['50_SMA_prev'] < results['200_SMA_prev']), 'Signal'] = "Rising"
    results.loc[(results['50_SMA'] < results['200_SMA']) & (results['50_SMA_prev'] > results['200_SMA_prev']), 'Signal'] = "Dropping"
    results['Break out'] = results['Close'].shift(-1) > results['Close'] * 1.03
    results['Break out bad'] = results['Close'].shift(-1) > results['Close'] * .97
    results['Rolling Max'] = results['Close'].rolling(window=180, min_periods=1).max()
    
    # If the current close price is equal to the rolling maximum, set 'Peak', otherwise set an empty string
    results['New High'] = results.apply(lambda row: 'Peak' if row['Close'] == row['Rolling Max'] else '', axis=1)
    
    # Drop the 'Rolling_Max' column as it's no longer needed
    # Print the last few elements where Signal is not 0.0 or Bound Limits is not 'Consolidating'
   # results = results.tail(30)
   # print(results[(results['Signal'] != 0.0) | (results['Bound Limits'] != 'Consolidating')])
    #print("Buy it")




# This function runs the above functions for all the given stocks
# Modify the runThat function to accept a stock symbol
def runThat(stoink,bk1,bd1,bk2,bd2,ubl,rise,cons,bo,ol,peak,rsig,rsib,macd,vol):
    
    print("\n", stoink)  # Print the stock being analyzed
    results = makeDF(stoink)  # Fetch historical data for the stock
    results['Rolling_Volume_Mean'] = results['Volume'].rolling(window=30).mean()
    # ... rest of your code ...

    mvCheck(results) # Check if the stock is moving or in consolidation phase
    check_crossover(results)
    results = calculate_stochastics(results)
        # shift(1) will give the previous row's data, as we're shifting data down by 1
    results['Opened_Lower'] = results['Open'] < results['Close'].shift(1)

    results['Bullish Confidence'] = results.apply(calculate_bull_confidence, axis=1,bk1=bk1,bd1=bd1, bk2=bk2,bd2=bd2,ubl=ubl,rise=rise,cons=cons,bo=bo,ol=ol,peak=peak,rsig=rsig,rsib=rsib,macd=macd,vol=vol)
    results['Bearish Confidence'] = results.apply(calculate_bear_confidence, axis=1,bk1=bk1,bd1=bd1, bk2=bk2,bd2=bd2,bo=bo,ol=ol,peak=peak,rsig=rsig,rsib=rsib,macd=macd,vol=vol,ubl=ubl,rise=rise)
   # bull_score = results['Bullish Confidence'].tail(7).sum()
   # bear_score = results['Bearish Confidence'].tail(7).sum()

    results['Total Bullish Score'] = results['Bullish Confidence'].rolling(30).sum()
    results['Total Bearish Score'] = results['Bearish Confidence'].rolling(30).sum()
    
    #  results['avgbear'].tail(7).mean()
        
       

    return(results)


def calculate_stochastics(results, high_col='High', low_col='Low', close_col='Close', n=7):
    stoch = StochasticOscillator(results[high_col], results[low_col], results[close_col], n)
    results['%K'] = stoch.stoch()
    results['%D'] = stoch.stoch_signal()
    return results

def calculate_bull_confidence(row,bk1,bd1,bk2,bd2,ubl,rise,cons,ol,bo,peak,rsig,rsib,macd,vol):
    score = 0
    # Stochastic Oscillator
    if row['%K'] < 20:  # Oversold territory
        score += bk1
    if row['%D'] < 20:  # Oversold territory
        score += bd1
    if row['%K'] > 80:  # Overbought territory
        score -= bk2
    if row['%D'] > 80:  # Overbought territory
        score -= bd2

    # Moving Averages and Price Action
    if row['Bound Limits'] == 'Risen above UBL':
        score += ubl
    if row['Signal'] == 'Rising':
        score += rise
    if row['Bound Limits'] == 'Consolidating':
        score += cons
    if row['Opened_Lower'] :
        score += ol
    if row['Break out'] == True:
        score += bo
    if row['New High'] == 'Peak':
        score -= peak
    if row['RSI'] < 30:
        score += rsig
    elif row['RSI'] > 70:
        score -= rsib

    # MACD
    if row['MACD'] > row['MACD_signal']:
        score += macd

    # Volume
    if row['Volume'] > row['Rolling_Volume_Mean']:
        score += vol

    return score

def calculate_bear_confidence(row,bk1,bd1,bk2,bd2,bo,ol,peak,rsig,rsib,macd,vol,ubl,rise):
    score = 0
    # Stochastic Oscillator
    if row['%K'] > 80:  # Overbought territory
        score += bk1
    if row['%D'] > 80:  # Overbought territory
        score += bd1
    if row['%K'] < 20:  # Oversold territory
        score -= bk2
    if row['%D'] < 20:  # Oversold territory
        score -= bd2

    # Moving Averages and Price Action
    if row['Bound Limits'] == 'Dropped below LBL':
        score += ubl
    if row['Signal'] == 'Dropping':
        score += rise
    if not row['Opened_Lower'] :
        score -= ol
    if row['Break out bad'] == True:
        score += bo
    if row['New High'] == 'Peak':
        score += peak
     # RSI
    if row['RSI'] > 70:
        score += rsig
    elif row['RSI'] < 30:
        score -= rsib

    # MACD
    if row['MACD'] < row['MACD_signal']:
        score += macd

    # Volume
    if row['Volume'] > row['Rolling_Volume_Mean']:
        score += vol
    return score



# In your runThat() function:


# Add the confidence level to the dataframe



# Run the script



# Usage





#for stoinks in stoink:
 #   results = (makeDF(stoink))
  #  mvCheck(results)
#print(df[:,'Low'].min())
#high = df.loc[0, 'High']
#low = df.loc[0, 'Low']
#print(df[['low']])
#print(avgHigh, avgLow)

################################################################




class Pas:
    def dynamic_trade_size(self, confidence):
        
        # Implement your logic here to determine trade size based on score and confidence
        if  1 ==1:
            #  print("high")
            return  .01 # 5% of available cash
        elif  confidence > 18:
            #print("medium")
            return  .10# 10% of available cash
        else:
            #  print("low")
            return .05 # 5% of available cash

    def __init__(self):
        self.data = None

    def init(self, data):    
        self.data = data
        self.bull_score = data['Total Bullish Score']
        self.bear_score = data['Total Bearish Score']
        self.bull_confidence = data['Bullish Confidence']
        self.bear_confidence = data['Bearish Confidence']

    def next(self, stoink):
        opt_vals = optimal_values.get(stoink, {})
        TPbuy = opt_vals.get('TPbuy', 1.01)
        TPsell = opt_vals.get('TPsell', 0.99)
        buyadd = opt_vals.get('buyadd', 1)
        selladd = opt_vals.get('selladd', 1)
        bk1 = opt_vals.get('bk1', 1)
        bd1 = opt_vals.get('bd1', 1)
        bk2 = opt_vals.get('bk2', 1)
        bd2 = opt_vals.get('bd2', 1)
        ubl = opt_vals.get('ubl', 1)
        rise = opt_vals.get('rise', 1)
        cons = opt_vals.get('cons', 1)
        bo = opt_vals.get('bo', 1)
        ol = opt_vals.get('ol', 1)
        peak = opt_vals.get('peak', 1)
        rsig = opt_vals.get('rsig', 1)
        rsib = opt_vals.get('rsib', 1)
        macd = opt_vals.get('macd', 1)
        vol = opt_vals.get('vol', 1)
        #high_confidence_threshold = opt_vals.get('high_confidence_threshold', 1)
        bull_score_today = self.bull_score.iloc[-1]
        bear_score_today = self.bear_score.iloc[-1]
        today_bull_score = self.bull_confidence.iloc[-1]
        today_bear_score = self.bear_confidence.iloc[-1]
        price = round(self.data['Close'].iloc[-1], 2)

        # Check if we have enough data for comparison
        if len(self.bull_score) > 1:
            cash = float(api.get_account().cash)
            #print(cash)
            trade_size_buy = int((cash * self.dynamic_trade_size(today_bull_score)) / price)
            trade_size_sell = int((cash * self.dynamic_trade_size(today_bear_score)) / price)
            #print(price)
            if bull_score_today > self.bull_score.iloc[-10] + buyadd and today_bull_score > self.bull_confidence.iloc[-10] + buyadd:
            
                #print(buyadd,stoink,TPbuy)
                
                api.submit_order(
                    symbol=stoink,
                    qty=trade_size_buy,
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=price,
                    order_class='bracket',
                    take_profit=dict(limit_price=round(price*1.6, 2)),  # Rounded to 2 decimal places
                    stop_loss=dict(stop_price=round(price*0.97, 2)),
                    #trail_percent= .5
                )
                print("bought long")
            elif bear_score_today > self.bear_score.iloc[-10] + selladd and today_bear_score > self.bear_confidence.iloc[-10] + selladd:
            
                api.submit_order(
                    symbol=stoink,
                    qty=trade_size_sell,  # This is the quantity you want to short
                    side='sell',  # 'sell' to short the stock
                    type='limit',
                    time_in_force='day',
                    limit_price=price,  # This is the price at which you want to initiate the short
                    order_class='bracket',
                    take_profit=dict(limit_price=round(price*.94, 2)),  # This should be lower than the current price
                    stop_loss=dict(stop_price=round(price*1.03, 2)),  # This should be higher than the current price
                   # trail_percent= .5
                    
                )
                print("bought short")
    # Infinite loop for paper trading
strategy = Pas()
for stoink in stoinks:
    try:  # Wrap the block of code with try-except to catch and continue on errors
    # Update the strategy with the latest data for the current stock
        latest_data = runThat(stoink,bk1,bd1,bk2,bd2,ubl,rise,cons,bo,ol,peak,rsig,rsib,macd,vol)
        strategy.init(latest_data)

        # Call the next method to decide on trading actions
        strategy.next(stoink)

        print(f"Processed {stoink}. Sleeping...")

    except Exception as e:  # Catch the exception and print an error message
        print(f"An error occurred while processing {stoink}: {e}")
        continue

# No need to sleep after all stocks are processed if you're running the loop only once
print("Completed one cycle for all stocks. Exiting.")