#TESTE  stock historical data
# You may need to install this librarys pip install yfinance and yahoo_fin:
import yfinance as yf  
from yahoo_fin import stock_info as si

datayahoo = si.get_data("TSLA", start_date="2023-04-01", end_date="2024-01-01")
datayahoo["Tomorrow"] = datayahoo["close"].shift(-1)  # Assuming 'close' is the column representing close prices
datayahoo["Target"] = (datayahoo["Tomorrow"] > datayahoo["close"]).astype(int) 
datayahoo = datayahoo.loc["1990-01-01":].copy()
print(datayahoo)
print(datayahoo.columns)

#datafinance = yf.download("AAPL", start="2023-04-01", end="2024-01-01")
#print(datafinance)
#print(datafinance.columns)

