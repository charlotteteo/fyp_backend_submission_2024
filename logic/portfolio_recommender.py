import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score,f1_score
import matplotlib.pyplot as plt
from arch import arch_model
import numpy as np
import pandas as pd
import yfinance as yf
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt import objective_functions
from pypfopt.black_litterman import BlackLittermanModel

# tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'INTC', 'IBM', 'CSCO', 'ORCL', 'AMD', 'TXN', 'QCOM']


from datetime import datetime, timedelta
def date_forward_calculator(start_date, shift):

    # Calculate the end date by adding the rolling window size in days
    end_date_dt = start_date + timedelta(days=shift)


    end_date = end_date_dt.strftime('%Y-%m-%d')

    return end_date

# Function to fetch historical stock prices
def fetch_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)[['Adj Close','Volume']]
    return stock_data


def calculate_ema(df, price_column, ema_column, alpha):
    df[ema_column] = df[price_column].ewm(alpha=alpha, adjust=False).mean()
    return df

def calculate_sma(df, price_column, sma_column, window=20):
    df[sma_column] = df[price_column].rolling(window=window).mean()
    return df

def calculate_ema(df, price_column, ema_column, alpha=0.2):
    df[ema_column] = df[price_column].ewm(alpha=alpha, adjust=False).mean()
    return df

def calculate_rsi(df, price_column, rsi_column, window=14):
    price_diff = df[price_column].diff(1)
    gain = price_diff.where(price_diff > 0, 0)
    loss = -price_diff.where(price_diff < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    df[rsi_column] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df, price_column, macd_column, signal_column, short_window=12, long_window=26, signal_window=9):
    short_ema = df[price_column].ewm(span=short_window, adjust=False).mean()
    long_ema = df[price_column].ewm(span=long_window, adjust=False).mean()
    
    df[macd_column] = short_ema - long_ema
    df[signal_column] = df[macd_column].ewm(span=signal_window, adjust=False).mean()
    
    return df

def calculate_bollinger_bands(df, price_column, bb_upper_column, bb_lower_column, window=20, num_std=2):
    rolling_mean = df[price_column].rolling(window=window).mean()
    rolling_std = df[price_column].rolling(window=window).std()
    
    df[bb_upper_column] = rolling_mean + (rolling_std * num_std)
    df[bb_lower_column] = rolling_mean - (rolling_std * num_std)
    
    return df
def calculate_top_k_percent(df, price_column, k):
    df['Top_K_Percent'] = df[price_column].pct_change().rolling(window=len(df), min_periods=1).sum().gt(k / 100).astype(int)
    return df

def calculate_obv(df, price_column, volume_column, obv_column='OBV'):
    df[obv_column] = (df[price_column].pct_change().shift(-1) > 0).astype(int) * df[volume_column]
    df[obv_column] = df[obv_column].cumsum()
    return df


# Assuming you have a DataFrame named df with 'Date' as the index
# Adjust the column names as needed

# Example usage:
# Assuming df is your DataFrame with 'Date' as the index and 'Stock_Price' as the stock price column
def get_tech_indicators(df,stock_name):
    df = calculate_ema(df, price_column=stock_name, ema_column=f"EMA_{stock_name}", alpha=0.2)
    stock_name = f"EMA_{stock_name}"
    df = calculate_sma(df, price_column=stock_name, sma_column='SMA_20', window=20)
    df = calculate_ema(df, price_column=stock_name, ema_column='EMA_12', alpha=0.2)
    df = calculate_rsi(df, price_column=stock_name, rsi_column='RSI_14')
    df = calculate_macd(df, price_column=stock_name, macd_column='MACD', signal_column='Signal', short_window=12, long_window=26, signal_window=9)
    df = calculate_bollinger_bands(df, price_column=stock_name, bb_upper_column='BB_Upper', bb_lower_column='BB_Lower', window=20, num_std=2)
    # Calculate Top K Percent (e.g., top 10%)
    # df = calculate_top_k_percent(df, stock_name, k=10)
    # Calculate On-Balance Volume (OBV)
    df = calculate_obv(df, stock_name, 'Volume')

    return df



def converter(dir):
    if dir:
        return 1
    else:
        return -1
        
def get_labelled_df(stock_data, m):
    df_overall = get_tech_indicators(stock_data, 'Adj Close')
    df_overall['Price_Direction'] = df_overall[ 'Adj Close'].shift(-m) > df_overall[ 'Adj Close']
    df_overall.dropna(inplace=True)
    df_overall['Price_Direction'] = df_overall['Price_Direction'].apply(lambda x: converter(x))
    # Drop rows with NaN in the label column
    df_overall = df_overall.reset_index()
    df_overall['forecast_date'] = df_overall['Date'].shift(-m)
    df_overall = df_overall.dropna()
    return df_overall

def get_actual_return(actual_price, last_price):
        return (actual_price - last_price)/last_price


def get_recommended_allocation(tickers,start_date,end_date,step_size=21):
    dict_cum = []
    dict_ref = {}
    dict_ref['stock'] = []
    dict_ref['last_price'] = []
    dict_ref['expected_price'] = []
    dict_ref['expected_return'] = []
    dict_ref['forecast_date'] = []
    for stock in tickers:
        dict_rec = {}
        og_stock_data = fetch_stock_data(stock, start_date, end_date)
        # df_overall = get_labelled_df(og_stock_data, step_size)

        og_stock_data
        get_tech_indicators(og_stock_data, 'Adj Close')
        forecast_data = get_tech_indicators(og_stock_data, 'Adj Close').iloc[-1:]
        forecast_data
        df_overall = get_labelled_df(og_stock_data, step_size)
        df_overall
        train_data, test_data = train_test_split(df_overall, test_size=0.2)
        X_train, y_train = train_data.drop(['forecast_date','Date','Price_Direction'], axis=1), train_data['Price_Direction']
        X_test, y_test = test_data.drop(['forecast_date','Date','Price_Direction'], axis=1), test_data['Price_Direction']
        # Initialize and train RandomForestClassifier
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
        # Predict on the test set
        predictions = rf_classifier.predict(X_test)
        # Assess accuracy
        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        stock_pred_direction = rf_classifier.predict(forecast_data)
        stock_pred_direction
        selected_data = og_stock_data.iloc[::-step_size][::-1]
        selected_data
        returns = np.log(selected_data[ 'Adj Close'].shift(-1) / selected_data[ 'Adj Close']).dropna()
        returns
        garch_model = arch_model(returns, vol='Garch', p=1, q=1)
        garch_result = garch_model.fit()

        # Forecast volatility for the next day
        forecasts = garch_result.forecast(horizon=1)
        forecasted_volatility = np.sqrt(forecasts.variance[-1:].values[0, 0])  # Sum over the forecast period

        # Modify based on your forecasting goals
        expected_return = returns.iloc[-1] + (forecasted_volatility * stock_pred_direction[0])
        expected_return

        # Calculate the expected price for the next day
        last_price = selected_data[ 'Adj Close'].iloc[-1]
        last_price
        expected_price = last_price * np.exp(expected_return)

        # Print the expected price
        print("Expected price 21 days from now:", expected_price)

        # expected_return = get_actual_return(expected_price, last_price)
        
        expected_return = get_actual_return(expected_price, last_price)
    

        forecast_date = date_forward_calculator(list(selected_data.reset_index()['Date'])[-1], 30)

        dict_ref['stock'] += [stock]
        dict_ref['last_price'] += [round(last_price,4)]
        dict_ref['expected_price'] += [round(expected_price,4)]
        dict_ref['expected_return'] += [round(expected_return,4)]
        dict_ref['forecast_date'] += [forecast_date]


        dict_rec['stock'] = stock
        dict_rec['last_price'] = round(last_price,2)
        dict_rec['expected_price']=round(expected_price,2)
        dict_rec['expected_return'] = [f"{round(expected_return*100,2)}%"]
        dict_rec['forecast_date'] = forecast_date
        dict_cum += [dict_rec]

    
    risk_averse_level = 0.5
    overall_exp_df = pd.DataFrame(dict_ref).dropna().set_index('stock')
    mu = overall_exp_df['expected_return']
    overall_stock_data = yf.download(list(pd.DataFrame(dict_ref)['stock']),start_date, end_date)['Adj Close']
    cov_matrix = risk_models.CovarianceShrinkage(overall_stock_data).ledoit_wolf()
    ef = EfficientFrontier(mu, cov_matrix)
    returns = overall_stock_data.pct_change().dropna()
    market_mu = returns.mean()

    viewdict = pd.DataFrame(dict_ref)[['stock','expected_return']].set_index('stock').to_dict()['expected_return']
    # Initialize the EfficientFrontier object with Black-Litterman model
    bl_model = BlackLittermanModel(cov_matrix, pi=market_mu,  absolute_views=viewdict)

    # Assuming P and Q vectors are defined somewhere

    bl_equilibrium_returns = bl_model.bl_returns()

    # Use the equilibrium returns in the EfficientFrontier optimization
    ef.expected_returns = bl_equilibrium_returns

    # Add constraints and maximize Sharpe ratio
    ef.add_constraint(lambda w: w >= 0)
    ef.add_constraint(lambda w: w <= 0.5)
    # ef.efficient_risk(risk_averse_level)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1) 
    # ef.max_sharpe()
    ef.efficient_risk(risk_averse_level)

    # Get cleaned weights
    cleaned_weights_bl = ef.clean_weights()
    labels = list(cleaned_weights_bl.keys())
    series = list(cleaned_weights_bl.values())


    pd.DataFrame(dict_ref)
    return {"expected_returns":dict_cum, "allocated_weights":
    {"labels":labels, "series": series}
    }






