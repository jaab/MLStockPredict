# Import necessary libraries
import logging
import numpy as np
import tkinter as tk
import threading
import datetime as dt
from datetime import datetime
import mplcursors  # Import the mplcursors library
import matplotlib.pyplot as plt
from tkcalendar import DateEntry  
from tkinter import ttk, messagebox
from pandas.plotting import register_matplotlib_converters
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn import metrics
from utils import predict_buy_sell,train_model,preprocess_data,create_features,handle_hover

# Configure the logging system
logging.basicConfig(filename='stock_predictions.log', level=logging.ERROR)

def plot_data(data, symbol):
    """
    Function Plot stock data, including closing prices, moving averages, and Buy/Sell signals.
    Parameters: data (pd.DataFrame): DataFrame containing stock data.
    Returns: None
    """
    try:
        #print("Data shape:", data.shape) 
        #print(data)
        last_price = data['close'].iloc[-1] # get the last price
        register_matplotlib_converters()
        figure = Figure(figsize=(12, 10), dpi=100)
        # Subplot 1: Closing prices with SMA_50, SMA_200, Bollinger Bands
        ax1 = figure.add_subplot(3, 1, 1)
        ax1.plot(data.index, data['close'], label='Close Price', linewidth=2)
        ax1.plot(data.index, data['SMA_50'], label='SMA_50',linestyle='--', linewidth=2)
        ax1.plot(data.index, data['SMA_200'], label='SMA_200', linestyle='--', linewidth=2)
        ax1.plot(data.index, data['Bollinger_Bands_Upper'], label='Upper Bollinger Band', linestyle='--', linewidth=2)
        ax1.plot(data.index, data['Bollinger_Bands_Lower'], label='Lower Bollinger Band', linestyle='--', linewidth=2)
        ax1.set_title(f'{symbol} Stock Analysis - Last Price: {last_price:.2f}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        # Customize tooltip content
        mplcursors.cursor([ax1], hover=True).connect("add", lambda sel: handle_hover(sel, symbol))
        # Subplot 2: Buy/Sell signals
        ax2 = figure.add_subplot(3, 1, 2, sharex=ax1)
        ax2.plot(data.index, data['close'], label='Close Price', linewidth=1)
        ax2.plot(data[data['Signal'] == 1].index, data['close'][data['Signal'] == 1], '^', markersize=3, color='g', label='Buy Signal')
        ax2.plot(data[data['Signal'] == 0].index, data['close'][data['Signal'] == 0], 'v', markersize=3, color='r', label='Sell Signal')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.legend()
        # Customize tooltip content
        mplcursors.cursor([ax2], hover=True).connect("add", lambda sel: handle_hover(sel, symbol))     
        # Subplot 3: Volume
        ax3 = figure.add_subplot(3, 1, 3, sharex=ax1)
        ax3.bar(data.index, data['volume'], color='gray', alpha=0.5)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volume')
        ax3.legend()
        # Customize tooltip content
        mplcursors.cursor([ax3], hover=True).connect("add", lambda sel: sel.annotation.set_text(
            f"Symbol: {symbol}\nVolume: {sel.target[1]:.2f}"))
        #Add sliders for customizable parameters (example: SMA windows)
        slider_ax = figure.add_axes((0.1, 0.02, 0.65, 0.03), facecolor='lightgoldenrodyellow')
        slider_short = Slider(slider_ax, 'SMA Short', 1, 100, valinit=50, valstep=1)
        slider_long = Slider(slider_ax, 'SMA Long', 1, 200, valinit=200, valstep=1)
        # Define update function for sliders
        def update(val):
            short_window = int(slider_short.val)
            long_window = int(slider_long.val)
            updated_data = create_features(data.copy(), shortlong_Label, short_window, long_window)
            plot_data(updated_data,symbol)
        # Connect sliders to update function
        slider_short.on_changed(update)
        slider_long.on_changed(update)
        # Add labels to sliders
        slider_ax.text(0.5, -0.15, 'Short Window', transform=slider_ax.transAxes, ha='center', fontsize=10)
        slider_ax.text(0.5, -0.35, 'Long Window', transform=slider_ax.transAxes, ha='center', fontsize=10)
        # Display the plots
        for widget in window.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()
    except Exception as e:
        status_label.config(text=f"Error during plot creation: {e}")

def evaluate_model(final_model, X_test, y_test,final_data,labels=None):
    """
    Function to evalute model.
    Parameters:final_model, X_test, y_test,final_data,labels
    Returns:None
    """
    try:   
        # Make predictions 
        predictions = final_model.predict(X_test)
         # Additional classification metrics
        roc_auc = roc_auc_score(y_test, predictions)
        pr_curve_precision, pr_curve_recall, _ = precision_recall_curve(y_test, predictions)
        pr_auc = auc(pr_curve_recall, pr_curve_precision)
        # Update the label with additional classification metrics
        additional_metrics_label.config(
            text=f"ROC-AUC: {roc_auc:.4f}\nPR AUC: {pr_auc:.4f}"
        )
        # Calculate and print classification report and confusion matrix
        report_classification = classification_report(y_test, predictions)
        # Update the label with the classification report
        CReport_Label.config(text=f"Classification Report:\n {report_classification}")
        # Calculate confusion matrix with labels parameter
        labels = [0, 1]  # Assuming you have binary classification
        report_matrix = confusion_matrix(y_test, predictions, labels=labels)
        # Update the label with the confusion matrix
        CMatrix_Label.config(text=f"Confusion Matrix:\n {report_matrix}")
        # Print accuracy and sentiment
        accuracy = metrics.accuracy_score(y_test, predictions.round())
        accuracy_label.config(text=f'Model Accuracy: {float(accuracy) * 100:.2f}')
        # Calculate the percentage of signals
        signal_percentage = (final_data['Signal'].sum() / len(final_data)) * 100
        # Determine the text 'Buy' or 'Sell'
        signal_text = 'Buy' if signal_percentage > 70 else ('Hold' if 30 < signal_percentage < 70 else 'Sell')
        # Update the prediction label
        sentiment_Label.config(text=f'AI Score: {(signal_percentage).astype(int):.0f} ({signal_text})')
        # Calculate precision, recall, and F1-score
        precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_test, predictions, labels=labels, average='binary')
        # Update the label with precision, recall, and F1-score
        PRF_Label.config(text=f"Precision: {precision:.2f},\n Recall: {recall:.2f},\n F1-Score: {f1_score:.2f}")
        # Check if predict_proba method is available and calculate confidence
        if hasattr(final_model, 'predict_proba'):
            # Get the predicted probabilities for the positive class
            if final_model.classes_.shape[0] == 2:  # Check if there are two classes
                #confidence = np.array(final_model.predict_proba(X_test))[:, 1].astype(float).tolist()
                confidence = np.array(final_model.predict_proba(X_test)[:, 1] * 100).astype(int).tolist()
            else:
                #confidence = np.array(final_model.predict_proba(X_test)).astype(float).tolist()
                confidence = np.array(final_model.predict_proba(X_test)* 100).astype(int).tolist()
            # Update the label with the confidence level
            confidence_label.config(text=f'Confidence Level: {np.mean(confidence):.0f}')
        else:
            confidence_label.config(text='Model does not support predict_proba')
    except IndexError as e:
            # Log the error
            logging.error(f"IndexError occurred: {e}")
            # Display error message to the user
            status_label.config(text=f"An unexpected error occurred: {e}")  
    except Exception as e:
            # Log the error
            logging.error(f"Unexpected error occurred: {e}")
            # Display a generic error message to the user
            status_label.config(text=f"An unexpected error occurred: {e}")  
            
def preprocess_and_analyze(symbol, start_date, end_date):
    """
    Function preprocess and analyze data 
    Parameters: symbol, start_date, end_date
    Returns:None
    """
    processed_data = preprocess_data(symbol, start_date, end_date, shortlong_Label)
    if 'Signal' not in processed_data.columns:
        status_label.config(text="Error in data processing. The 'Signal' column is missing.")
        return
    if processed_data is not None:
        final_data = predict_buy_sell(processed_data)
        X = final_data[['SMA_50', 'SMA_200', 'Daily_Return', 'RSI_14', 'MACD_Line_12_26', 'Signal_Line_12_26', 'MACD_Histogram_12_26','MACD_Line_6_13','Signal_Line_6_13','MACD_Histogram_6_13']]
        y = final_data['Signal']
        if len(X) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            final_model = train_model(X_train, y_train, decisionTreeCross_Label, randomForestCross_Label, gradientBoostingCross_Label)
            evaluate_model(final_model, X_test, y_test, final_data, labels=None)
            # Update status label for model training
            status_label.config(text="Analysis Complete")
            plot_data(final_data, symbol)  
        else:
            status_label.config(text="Insufficient data for analysis.")
               
def fetch_and_analyze_data(symbol, start_date, end_date):
    """
    Function fetch and analyze_data 
    Parameters: symbol, start_date, end_date
    Returns:None
    """
    try:
        # Update status label for model training
        status_label.config(text="Model Training. Please wait...")
        processed_data = preprocess_data(symbol, start_date, end_date, shortlong_Label)
        if processed_data is not None:
            preprocess_and_analyze(symbol, start_date, end_date)
        else:
            status_label.config(text="Error in data processing.")
    except Exception as e:
        # Display a generic error message to the user
        status_label.config(text=f"An unexpected error occurred: {e}")   
    finally:
        # Reset the status label text when the analysis is complete or if an error occurs
        status_label.config(text="")
                
def analyze_stock():
    """
    Function Analyze stock data and display analysis results.
    Parameters:None
    Returns:None
    """
    global investment_horizon_label
    symbol = symbol_var.get()
    start_date = start_date_var.get()
    end_date = end_date_var.get()
    # Validate input fields
    if not symbol or not start_date or not end_date:
        status_label.config(text="Please fill in all input fields.")
        return
     # Validate stock symbol format
    if not symbol.isalpha():
        status_label.config(text="Invalid stock symbol. Please enter alphabetic characters only.")
        return
    # Validate date format
    try:
        dt.datetime.strptime(start_date, '%Y-%m-%d')
        dt.datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        status_label.config(text="Invalid date format. Please use 'YYYY-MM-DD' format.")
        return
    # Inform user about data fetching
    status_label.config(text="Fetching data. Please wait...")
    # Use threading to run data fetching and analysis concurrently
    threading.Thread(target=fetch_and_analyze_data, args=(symbol, start_date, end_date)).start()

# Create the main window
window = tk.Tk()
window.title("Stock Analysis")
# Create ttk.Style
style = ttk.Style()
# Configure the style for buttons
style.configure('TButton', font=('Helvetica', 12), padding=5)
style.configure('TLabel', font=('Helvetica', 12))
# Create and pack frames
input_frame = ttk.Frame(window, padding="10")
input_frame.pack(side=tk.LEFT, padx=10, pady=10)
output_frame_R = ttk.Frame(window)
output_frame_R.pack(side=tk.RIGHT, padx=10, pady=10)
output_frame_B = ttk.Frame(window)
output_frame_B.pack(side=tk.BOTTOM, padx=10, pady=10)
# Stock Symbol Input
symbol_label = ttk.Label(input_frame, text="Stock Symbol:")
symbol_label.grid(column=0, row=0, sticky=tk.W, padx=(0, 5), pady=(0, 5))
symbol_var = tk.StringVar()
symbol_entry = ttk.Entry(input_frame, textvariable=symbol_var)
symbol_entry.grid(column=1, row=0, sticky=tk.W, padx=(0, 5), pady=(0, 5))
# Start Date Input with Date Picker
start_date_label = ttk.Label(input_frame, text="Start Date:")
start_date_label.grid(column=0, row=1, sticky=tk.W, padx=(0, 5), pady=(0, 5))
start_date_var = tk.StringVar()
start_date_entry = DateEntry(input_frame, textvariable=start_date_var, date_pattern='yyyy-mm-dd')
start_date_entry.grid(column=1, row=1, sticky=tk.W, padx=(0, 5), pady=(0, 5))
# End Date Input with Date Picker
end_date_label = ttk.Label(input_frame, text="End Date:")
end_date_label.grid(column=0, row=2, sticky=tk.W, padx=(0, 5), pady=(0, 5))
end_date_var = tk.StringVar()
end_date_entry = DateEntry(input_frame, textvariable=end_date_var, date_pattern='yyyy-mm-dd')
end_date_entry.grid(column=1, row=2, sticky=tk.W, padx=(0, 5), pady=(0, 5))
# Analyze Button
analyze_button = ttk.Button(input_frame, text="Analyze", command=analyze_stock)
analyze_button.grid(column=0, row=5, columnspan=2, pady=(10, 0))
# Apply a custom style to the Analyze Button
style.configure('TButton', foreground='white', background='green', padding=10)
# Status Label for progress indication
status_label = ttk.Label(window, text="")
status_label.pack()
# Model Accuracy Label
accuracy_label = ttk.Label(output_frame_R, text="Model Accuracy: N/A")
accuracy_label.pack()
# Apply a custom style to labels in the output_frame_R
style.configure('TLabel', foreground='black')
# Sentiment Label
sentiment_Label = ttk.Label(output_frame_R, text="AI Score: N/A")
sentiment_Label.pack()
# Confidence Label
confidence_label = ttk.Label(output_frame_R, text="Confidence: N/A")
confidence_label.pack()
# Short Term and Long Term Label
shortlong_Label = ttk.Label(output_frame_R, text="Short/Long Term: N/A")
shortlong_Label.pack()
# Precision, Recall, and F1-score Label
PRF_Label = ttk.Label(output_frame_R, text="PRF: N/A")
PRF_Label.pack()
# Confusion Matrix results Grid
CMatrix_Label = ttk.Label(output_frame_B, text="Confusion Matrix:\n")
CMatrix_Label.grid(column=0, row=0)
# Classification Report results Grid
CReport_Label = ttk.Label(output_frame_B, text="Classification Report:\n")
CReport_Label.grid(column=1, row=0)
# Additional Metrics Label
additional_metrics_label = ttk.Label(output_frame_R, text="Additional Metrics:\n")
additional_metrics_label.pack()
decisionTreeCross_Label = ttk.Label(output_frame_R, text="Decision Tree Cross Accuracy: N/A")
decisionTreeCross_Label.pack()
randomForestCross_Label = ttk.Label(output_frame_R, text="Random Forest Cross Accuracy: N/A")
randomForestCross_Label.pack()
gradientBoostingCross_Label = ttk.Label(output_frame_R, text="Gradient Boosting Cross Accuracy: N/A")
gradientBoostingCross_Label.pack()
# Run the Tkinter event loop
window.mainloop()
