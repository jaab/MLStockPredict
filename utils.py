#Functions for fetching historical stock data, creating features, train_model, predict_buy_sell, preprocess_data, calculate_rsi and calculate_macd.
import numpy as np
import datetime as dt
from datetime import datetime
from tkinter import messagebox
from yahoo_fin import stock_info as si
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Function to fetch historical stock data using yahoo_fin
def get_yahoo_fin_data(symbol, start_date, end_date):
    try:
        data = si.get_data(symbol, start_date=start_date, end_date=end_date)
        data["tomorrow"] = data["close"].shift(-1)  
        data["target"] = (data["tomorrow"] > data["close"]).astype(int) 
        data = data.loc["1990-01-01":].copy()
        return data
    except AssertionError as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Function to calculate moving averages
def calculate_moving_averages(data, short_window, long_window):
    data['SMA_50'] = data['close'].rolling(window=short_window).mean()
    data['SMA_200'] = data['close'].rolling(window=long_window).mean() 
    data['Daily_Return'] = data['close'].pct_change() 
    # Add your crossover strategy logic here SMA or EMA
    #data['SMA_Short'] = data['close'].rolling(window=short_window).mean()
    #data['SMA_Long'] = data['close'].rolling(window=long_window).mean()
    data['EMA_Short'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_Long'] = data['close'].ewm(span=long_window, adjust=False).mean()

# Function to calculate additional indicators
def calculate_additional_indicators(data, rsi_window, macd_short_12_26, macd_long_12_26, macd_short_6_13, macd_long_6_13):
    data['Bollinger_Bands_Upper'], data['Bollinger_Bands_Lower'] = calculate_bollinger_bands(data['close'], window=20)
    data['ATR'] = calculate_average_true_range(data['high'], data['low'], data['close'], window=14)
    data['RSI_14'] = calculate_rsi(data['close'], window=rsi_window)
    data['MACD_Line_12_26'], data['Signal_Line_12_26'], data['MACD_Histogram_12_26'] = calculate_macd(data['close'], short_window=macd_short_12_26, long_window=macd_long_12_26)
    data['MACD_Line_6_13'], data['Signal_Line_6_13'], data['MACD_Histogram_6_13'] = calculate_macd(data['close'], short_window=macd_short_6_13, long_window=macd_long_6_13)

# Function to identify crossover pointS   
def identify_crossover_points(data):
    if 'SMA_Short' in data.columns and 'SMA_Long' in data.columns:
        # Calculate crossover signal based on SMA
        data['Crossover'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1, 0)
        data['Crossover_Signal'] = data['Crossover'].diff()
    else:
        # Calculate crossover signal based on EMA
        # Calculate the difference between short-term and long-term EMAs 
        data['EMA_Diff'] = data['EMA_Short'] - data['EMA_Long']
        # Introduce a threshold for crossover signals
        threshold = 0.01
        data['Crossover_Signal'] = np.where(data['EMA_Diff'] > threshold, 1, np.where(data['EMA_Diff'] < -threshold, -1, 0))

# Function to identify trends
def identify_trends(data):
    # Identify trend directions
    if 'SMA_Short' in data.columns and 'SMA_Long' in data.columns:
        data['Short_Trend'] = np.where(data['SMA_Short'] > data['SMA_Short'].shift(1), 'Up', 'Down')
        data['Long_Trend'] = np.where(data['SMA_Long'] > data['SMA_Long'].shift(1), 'Up', 'Down')
    else:
        data['Short_Trend'] = np.where(data['EMA_Short'] > data['EMA_Short'].shift(1), 'Up', 'Down')
        data['Long_Trend'] = np.where(data['EMA_Long'] > data['EMA_Long'].shift(1), 'Up', 'Down')

# Function to update investment horizon label
def update_investment_horizon_label(data, shortlong_Label):
    shortlong_Label.config(text=f"\nInvestment Horizon Details:\n"
                                f"Last Crossover Signal: {data['Crossover_Signal'].iloc[-1]}\n"
                                f"Short-Term Trend: {data['Short_Trend'].iloc[-1]}\n"
                                f"Long-Term Trend: {data['Long_Trend'].iloc[-1]}\n\n")

# Function update data horizons
def update_data_horizons(data):
    horizons = [2, 5, 60, 200]
    #horizons = [2, 5, 250, 1000]
    new_predictors = []
    for horizon in horizons:
        rolling_averages = data['close'].rolling(horizon).mean()
        ratio_column = f"close_ratio_{horizon}"
        data[ratio_column] = data["close"] / rolling_averages
        trend_column = f"trend_{horizon}"
        data[trend_column] = data["target"].shift(1).rolling(horizon).sum()
        new_predictors += [ratio_column, trend_column]
    return data

# Function to predict Buy (1) or Sell (0)
def predict_buy_sell(data):
    if data.empty:
        messagebox.showerror("Data Error", "No data available for the specified time range.")
        return data
    data.loc[:, 'Signal'] = 0  # 0 represents 'Sell'
    data.loc[data['SMA_50'] > data['SMA_200'], 'Signal'] = 1  # 1 represents 'Buy'
    return data

# Function to Combine signals from different indicators
def combine_signals(data):
    data['Combined_Signal'] = np.where(
        (data['Crossover_Signal'] == 1) & (data['RSI_14'] < 70) & (data['MACD_Histogram_12_26'] > 0),
        'Buy',
        np.where(
            (data['Crossover_Signal'] == -1) & (data['RSI_14'] > 30) & (data['MACD_Histogram_12_26'] < 0),
            'Sell',
            'Hold'
        )
    )

# Function create features
def create_features(data, shortlong_Label, short_window=50, long_window=200, rsi_window=14, macd_short_12_26=12, macd_long_12_26=26, macd_short_6_13=6, macd_long_6_13=13):
    calculate_moving_averages(data, short_window, long_window)
    calculate_additional_indicators(data, rsi_window, macd_short_12_26, macd_long_12_26, macd_short_6_13, macd_long_6_13)
    identify_crossover_points(data)
    identify_trends(data)
    combine_signals(data)  
    data = update_data_horizons(data)
    # Use forward-fill to fill missing values
    data = data.ffill().dropna(subset=data.columns[data.columns != "tomorrow"])
    # Use backward-fill to fill missing values
    #data = data.bfill().dropna(subset=data.columns[data.columns != "tomorrow"])
    # Add 'Signal' column based on buy/sell logic
    data = predict_buy_sell(data)
    # Instantiate and use update_data_horizons
    update_investment_horizon_label(data, shortlong_Label)
    return data

# Function to process data
def preprocess_data(symbol, start_date, end_date, shortlong_Label):
    """
    Function to preprocess stock data.
    Parameters: symbol (str), start_date (str), end_date (str), shortlong_Label (tk.Label)
    Returns: pd.DataFrame
    """
    stock_data = get_yahoo_fin_data(symbol, start_date, end_date)
    # Check if stock_data is available
    if stock_data is None:
        messagebox.showerror("Data Error", "No data available for the specified time range.")
        return None
    # Check if there is enough data for splitting
    if len(stock_data) < 2:
        messagebox.showerror("Data Error", "Insufficient data for analysis.")
        return None
    processed_data = create_features(stock_data, shortlong_Label)
    return processed_data

def train_model(X_train, y_train,decisionTreeCross_Label,randomForestCross_Label,gradientBoostingCross_Label):
    """
    Function to train model.
    Parameters: X_train, y_train
    Returns: final_model
    """
    # Create and train the Decision Tree model
    #decision_tree_model = DecisionTreeClassifier(random_state=42)
    #decision_tree_model.fit(X_train, y_train)
    # Create and train the Random Forest model
    random_forest_model = RandomForestClassifier(random_state=42)
    random_forest_model.fit(X_train, y_train)
    # Create and train the Gradient Boosting model
    gradient_boosting_model = GradientBoostingClassifier(random_state=42)
    gradient_boosting_model.fit(X_train, y_train)
      # **********************Perform Grid Search for hyperparameter tuning*************************
    param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(random_forest_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    # Get the best hyperparameters from the grid search
    best_max_depth = grid_search.best_params_['max_depth']
    best_min_samples_split = grid_search.best_params_['min_samples_split']
    # Create and train the Decision Tree model
    decision_tree_model = DecisionTreeClassifier(random_state=42, max_depth=best_max_depth, min_samples_split=best_min_samples_split)
    decision_tree_model.fit(X_train, y_train)
    # Use cross-validation to evaluate models
    dt_cv_score = cross_val_score(decision_tree_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    rf_cv_score = cross_val_score(random_forest_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    gb_cv_score = cross_val_score(gradient_boosting_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    decisionTreeCross_Label.config(text=f'Decision Tree Cross Accuracy: {dt_cv_score:.4f}')
    randomForestCross_Label.config(text=f'Random Forest Cross Accuracy: {rf_cv_score:.4f}')
    gradientBoostingCross_Label.config(text=f'Gradient Boosting Accuracy: {gb_cv_score:.4f}')
    # Use the best-performing model for final training
    best_model = max([(decision_tree_model, dt_cv_score), (random_forest_model, rf_cv_score), (gradient_boosting_model, gb_cv_score)], key=lambda x: x[1])[0]
    best_model.fit(X_train, y_train)
    return best_model

#Function Calculate bollinger bands
def calculate_bollinger_bands(close_prices, window=20, num_std_dev=2):
    rolling_mean = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()
    upper_band = rolling_mean + (num_std_dev * rolling_std)
    lower_band = rolling_mean - (num_std_dev * rolling_std)
    return upper_band, lower_band

#Function Calculate average true range
def calculate_average_true_range(high_prices, low_prices, close_prices, window=14):
    high_low_diff = high_prices - low_prices
    high_close_diff = np.abs(high_prices - close_prices.shift(1))
    low_close_diff = np.abs(low_prices - close_prices.shift(1))
    true_range = np.maximum(high_low_diff, np.maximum(high_close_diff, low_close_diff))
    average_true_range = true_range.rolling(window=window).mean()
    return average_true_range

#Function Calculate RSI (Relative Strength Index):
def calculate_rsi(close_prices, window=14):
    # Calculate daily price changes
    price_diff = close_prices.diff(1)
    # Calculate gain (positive changes) and loss (negative changes)
    gain = price_diff.where(price_diff > 0, 0)
    loss = -price_diff.where(price_diff < 0, 0)
    # Calculate average gain and average loss over the specified window
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    # Calculate the Relative Strength (RS) and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

#Function to Calculate MACD (Moving Average Convergence Divergence):
def calculate_macd(close_prices, short_window=12, long_window=26):
    # Calculate short-term Exponential Moving Average (EMA)
    short_ema = close_prices.ewm(span=short_window, adjust=False).mean()
    # Calculate long-term Exponential Moving Average (EMA)
    long_ema = close_prices.ewm(span=long_window, adjust=False).mean()
    # Calculate MACD line
    macd_line = short_ema - long_ema
    # Calculate Signal line (9-day EMA of the MACD line)
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    # Calculate MACD Histogram
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

# function to create tooltip content
def handle_hover(sel, symbol):
    try:
        # Assuming sel.target[0] is a numpy.float64 representing a date
        date_float64 = sel.target[0]
        #print("date_float64:", date_float64)
        # Check if date_float64 is a valid numpy.float64
        if not np.issubdtype(type(date_float64), np.floating):
            raise ValueError("Invalid date format")
         # Convert to datetime
        date_datetime = dt.datetime.utcfromtimestamp(0) + dt.timedelta(days=date_float64)
        formatted_date = date_datetime.strftime('%Y-%m-%d %H:%M:%S')
        #print("Converted date:", formatted_date)
        # Customize tooltip content
        sel.annotation.set_text(f"Symbol: {symbol}\nPrice: {sel.target[1]:.2f}\nDate: {formatted_date}")
    except Exception as e:
        print(f"Error in handle_hover: {e}")
        raise e


