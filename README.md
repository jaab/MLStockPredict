# MLStockPredict

## Overview

The ML Stock Predict is designed for analyzing historical stock data, making predictions, and visualizing the results. It includes a graphical user interface (GUI) built using Tkinter for easy interaction. It allows users to analyze historical stock data, visualize trends, and make buy/sell predictions based on moving averages.

## Features

- Fetches historical stock data from Yahoo Finance using `yahoo_fin`.
- Implements various technical indicators such as Simple Moving Averages (SMA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD).
- Predicts Buy (1) or Sell (0) signals based on the technical indicators.
- Allows users to analyze stocks by inputting stock symbols and date ranges.
- Utilizes machine learning algorithms such as Decision Tree Classifier, Random Forest Classifier and Gradient Boosting Classifier for signal prediction.
- Performs hyperparameter tuning using Grid Search to optimize model performance.
- Displays accuracy, sentiment, classification report, and confusion matrix in the GUI.
- Visualize closing prices, moving averages, and buy/sell signals.
- Train a machine learning model for buy/sell predictions.
- Evaluate the model's accuracy and additional classification metrics.

## Machine Learning Traning Models
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier

## Software Requirements

- Python 3.x
- Required Libraries: numpy, tkinter, tkcalendar, pandas, matplotlib, sklearn, yahoo_fin,...

## IDE

The project can be developed and run using any Python-compatible IDE. Some popular choices include:

- Visual Studio Code
- PyCharm
- Jupyter Notebooks

## Startup 🚀

0. Clone the repository:
   ```bash
   git clone https://github.com/jaab/MLStockPredict.git

1. Install required Python libraries:
   pip install tkinter pandas yahoo_fin matplotlib scikit-learn,...

2. Run the index.py file:
    python index.py

3. Input stock symbol, start date, end date in the GUI.

4. Click the "Analyze" button to perform stock analysis.

5. View the results, including accuracy, AI Score sentiment, and metrics tables.

## Files and Directories

index.py: Main script for the GUI and stock analysis.
utils.py: Utility functions for fetching data, creating features, and model evaluation,...

## Additional Notes
This project uses Yahoo Finance API for fetching historical stock data.
Make sure to have an active internet connection for data fetching.

## Contributing
If you'd like to contribute to this project, please follow the standard GitHub workflow: fork the repository, create a new branch, make changes, and submit a pull request.

Feel free to customize the content of the README file based on your project's specific details and features. The goal is to provide clear and concise information to help users and contributors understand how to use and contribute to your application.


## Who, When, Why?
👨🏾‍💻 Author: Jaab
📅 Version: 1.x
📜 License: This project is licensed under the MIT License
