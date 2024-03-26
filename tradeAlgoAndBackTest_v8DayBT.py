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
from sklearn.ensemble import ExtraTreesRegressor

from skopt import gp_minimize
from skopt.plots import plot_objective
from skopt.callbacks import DeltaYStopper

# Define parameter space



# Incorrect
# regressor = ExtraTreesRegressor(criterion='mse')

# Corrected


# List of stocks to analyze
stoinks = ['HD','AAPL','COST','ACN','MCD','TMO','TMUS','PM','MS','VZ','NEE','BX','BMY','SPY','BIIB','MRNA','DFS','PTON','DXCM','MSCI','CRWD','HLT','IDXX','SQ','ROK','DASH','DDOG','SPOT','BKR','HAL','IDXX','SGEN','LNG','FICO','XYL','SPLK','HEI','ICLR','NET','RMD','ZM','BRO','ANSS','IR','PLTR','GWW','WST','PWR','VNC','ANSS','MLM','IR','MDB','HZNB','ZS','XYL','SPY', 'META', 'VLO', 'VMW', 'GOOG', 'MSFT', 'TSLA', 'T', 'ADNT', 'NFLX', 'EBAY', 'AAPL', 'UPS', 'AMZN', 'ADBE', 'PEP', 'PLD','NVDA','LLY','V','TSM','WMT','JPM','NVO','MA','ORCL','CVX','BABA','BAC','ACN','LIN','ABT','SAP','TMUS','HDB','NKE','DIS','TTE','WFC','UPS','VZ','CAT','INTC','MCD','BDX','CB','ABNB','SLB','CVS','ZTS','VRTX','REGN','LRCX','ETN','ADI','UBER','GILD','MMC','MDLZ','BKNG']
sSMA=2
lSMA =3
bk1=1
bd1=1
bk2=2
bd2=2
ubl=1
rise=1
cons=1
bo=1
ol=1
peak=1
rsig=1
rsib=1
macd=1
vol=1
results_dict = {}
# Initialize an empty dictionary to hold the results.
results_dict = {}

# Define the keys you're interested in.
keys_of_interest = ['TPbuy', 'TPsell', 'buyadd', 'selladd', 'bk1', 'bd1', 'bk2', 'bd2', 'ubl', 'rise', 'cons', 'bo', 'ol', 'peak', 'rsig', 'rsib','macd','vol']



# This function fetches 30-day historical data for the given stock 
def makeDF(stoink,sSMA,lSMA):
    current = yf.Ticker(stoink) # Get ticker data for the given stock
    hist = current.history(period='1mo',interval='5m') # Fetch 30-day historical data
    # Creating a DataFrame from the historical data
    df = hist
   
    df.reset_index(inplace=True) 
    df['50_SMA'] = df['Close'].rolling(window=sSMA).mean()# Reset index for the DataFrame
    df['200_SMA'] = df['Close'].rolling(window=lSMA).mean()
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
    avgHigh = results['High'].rolling(window=30).mean()
    avgLow = results['Low'].rolling(window=30).mean()
    
    
    # Shift the averages to get the average of the last 30 days on the next day
    avgHigh = avgHigh.shift(1)
    avgLow = avgLow.shift(1)
    
    # Calculate the UBL and LBL for each day
    UBL = avgHigh * 1.03
    LBL = avgLow * .97
    
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
    results['Break out'] = results['Close'].shift(-1) > results['Close'] * 1.01
    results['Break out bad'] = results['Close'].shift(-1) > results['Close'] * .99
    results['Rolling Max'] = results['Close'].rolling(window=90, min_periods=1).max()
    
    # If the current close price is equal to the rolling maximum, set 'Peak', otherwise set an empty string
    results['New High'] = results.apply(lambda row: 'Peak' if row['Close'] == row['Rolling Max'] else '', axis=1)
    
    # Drop the 'Rolling_Max' column as it's no longer needed
    # Print the last few elements where Signal is not 0.0 or Bound Limits is not 'Consolidating'
   # results = results.tail(30)
   # print(results[(results['Signal'] != 0.0) | (results['Bound Limits'] != 'Consolidating')])
    #print("Buy it")




# This function runs the above functions for all the given stocks
# Modify the runThat function to accept a stock symbol
def runThat(stoink,bk1,bd1,bk2,bd2,ubl,rise,cons,bo,ol,peak,rsig,rsib,macd,vol,sSMA,lSMA):
    
    print("\n", stoink)  # Print the stock being analyzed
    results = makeDF(stoink,sSMA,lSMA)  # Fetch historical data for the stock
    results['Rolling_Volume_Mean'] = results['Volume'].rolling(window=14).mean()
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

    results['Total Bullish Score'] = results['Bullish Confidence'].rolling(14).sum()
    results['Total Bearish Score'] = results['Bearish Confidence'].rolling(14).sum()
    
    #  results['avgbear'].tail(7).mean()
        
       

    return(results)


def calculate_stochastics(results, high_col='High', low_col='Low', close_col='Close', n=30):
    stoch = StochasticOscillator(results[high_col], results[low_col], results[close_col], n)
    results['%K'] = stoch.stoch()
    results['%D'] = stoch.stoch_signal()
    return results

def calculate_bull_confidence(row,bk1,bd1,bk2,bd2,ubl,rise,cons,ol,bo,peak,rsig,rsib,macd,vol):
    score = 0
    # Stochastic Oscillator
    if row['%K'] < 30:  # Oversold territory
        score += bk1
    if row['%D'] < 30:  # Oversold territory
        score += bd1
    if row['%K'] > 70:  # Overbought territory
        score -= bk2
    if row['%D'] > 70:  # Overbought territory
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
    if row['%K'] > 70:  # Overbought territory
        score += bk1
    if row['%D'] > 70:  # Overbought territory
        score += bd1
    if row['%K'] < 30:  # Oversold territory
        score -= bk2
    if row['%D'] < 30:  # Oversold territory
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


class SingleStockStrategy(Strategy):
    TPbuy=1.01
    buyadd= 1
    TPsell = .90
    selladd= 1
    high_confidence_threshold = 27  # New parameter
    medium_confidence_threshold = 10  # New parameter
    high_trade_size = 0.20  # New parameter
    medium_trade_size = 0.10  # New parameter
    low_trade_size = 0.05  # New parameter
    sSMA=1
    lSMA=2
    bk1=1
    bd1=1
    bk2=2
    bd2=2
    ubl=1
    rise=1
    cons=1
    bo=1
    ol=1
    peak=1
    rsig=1
    rsib=1
    macd=1
    vol=1
    def init(self):
        self.bull_score = self.I(lambda: self.data['Total Bullish Score'])
        self.bear_score = self.I(lambda: self.data['Total Bearish Score'])
        self.bull_confidence = self.I(lambda: self.data['Bullish Confidence'])
        self.bear_confidence = self.I(lambda: self.data['Bearish Confidence'])

    def next(self):
        bull_score_today = self.bull_score[-1]
        bear_score_today = self.bear_score[-1]
        today_bull_score = self.bull_confidence[-1]
        today_bear_score = self.bear_confidence[-1]
        price = self.data.Close[-1]
        cash = self.equity
        
       

        
        
        

        # Check if we have enough data for comparison
        if len(self.bull_score) > 1:
            if bull_score_today  > self.bull_score[-5] + self.buyadd and today_bull_score  > self.bull_confidence[-5] + self.buyadd:
           #     trade_size_buy = self.dynamic_trade_size(bull_score_today, today_bull_score)
               # print(bull_score_today)
               # size_buy = int(cash * trade_size_buy / price)
              #  print(f"Cash: {cash}, Price: {price}, Trade Size Buy: {trade_size_buy}, Size Buy: {size_buy}")

                
               # print(f"Cash: {cash}, Trade Size Buy: {trade_size_buy}, Size Buy: {size_buy}")
                self.buy(size=.1,sl=price*0.96, tp=price * 1.06)
            
            elif bear_score_today > self.bear_score[-5] + self.selladd and today_bear_score  > self.bear_confidence[-5] + self.selladd:
         #       trade_size_sell = self.dynamic_trade_size(bear_score_today, today_bear_score)
                
               # print(f"Cash: {cash}, Price: {price}, Trade Size sell: {trade_size_sell}, Size sell: {size_sell}")

                self.sell(size=.1,sl=price * 1.04, tp=price * .96)



            

   # def dynamic_trade_size(self, score, confidence):
      #  if confidence > 1:
       #     return .03  # now a parameter
        #elif confidence > 19:
         #   return .10  # now a parameter
        #else:
         #   return .5  # now a parameter


# Collect equity curves for each stock
equity_curves = []
total_trade_duration = 0
total_trades = 0
stats = []
all_data = {}

        

def objective(params):
    total_profit_factor = 0  # Initialize total profit factor
    successful_stocks = 0  # Count of stocks that did not produce an error

    for stoink in stoinks:
        try:
            # Unpack parameters
            buyadd, selladd, bk1, bd1, bk2, bd2, ubl, rise, cons, bo, ol, peak, rsig, rsib, macd, vol, sSMA, lSMA = params  
            # Run the strategy
            data = runThat(stoink, bk1, bd1, bk2, bd2, ubl, rise, cons, bo, ol, peak, rsig, rsib, macd, vol, sSMA, lSMA)

            data.index = pd.to_datetime(data.index)
            data.fillna(method='ffill', inplace=True)
            bt = Backtest(data, SingleStockStrategy, cash=100000)
            stats = bt.run(buyadd=buyadd, selladd=selladd, bk1=bk1, bk2=bk2, bd1=bd1, bd2=bd2, ubl=ubl, rise=rise, cons=cons, bo=bo, ol=ol, peak=peak, rsig=rsig, rsib=rsib, macd=macd, vol=vol, sSMA=sSMA, lSMA=lSMA)
            print(stats)

            if pd.isna(stats['Profit Factor']):
                continue  # Skip this stock if the profit factor is NaN
            
            total_profit_factor += stats['Profit Factor']
            successful_stocks += 1

        except Exception as e:  # Catch the exception and print an error message
            print(f"An error occurred while processing {stoink}: {e}")
            continue

    if successful_stocks == 0:  # Avoid division by zero
        return 1e10

    avg_profit_factor = total_profit_factor / successful_stocks
    return -avg_profit_factor

space = [
    (1, 5),  # buyadd
    (1, 5),  # selladd
    (1, 5),  # bk1
    (1, 5),  # buyadd
    (1, 5),  # selladd
    (1, 5),
    (1, 5),  # buyadd
    (1, 5),  # selladd
    (1, 5),
    (1, 5),  # buyadd
    (1, 5),  # selladd
    (1, 5),
    (1, 5),  # buyadd
    (1, 5),  # selladd
    (1, 5),
    (1, 5),
    (10, 20),
    (30, 90),
    #AAPL
#[5, 1, 1, 2, 5, 2, 5, 4, 3, 2, 1, 3, 3, 4, 5, 5, 18, 30]
#Best score: 1.4883830086822565
#[2, 4, 1, 3, 1, 5, 2, 4, 4, 2, 3, 3, 2, 2, 4, 2, 12, 61]
    ]

callback = DeltaYStopper(delta=0.01, n_best=5)

result = gp_minimize(
    objective, 
    space, 
    n_calls=15,
    n_initial_points=10,
    n_jobs=-1, 
    verbose=True,
    xi=0.1,
    kappa=8,
    callback=callback
)

best_parameters = result.x
print(best_parameters)
print("Best score:", -result.fun)
#_ = plot_objective(result, dimensions=['buyadd', 'selladd', 'bk1', 'bk2', 'bd1', 'bd2', 'ubl', 'rise', 'cons', 'bo', 'ol', 'peak', 'rsig', 'rsib', 'macd', 'vol', 'lSMA', 'sSMA'])
#plt.show()
