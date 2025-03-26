import yfinance as yf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os

def data_acquisition(ticker, start_date, end_date, interval="1d", save_path=None):
    """
    Download stock data using yfinance
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        interval (str): Data interval (1d, 1h, etc.)
        save_path (str, optional): Path to save the downloaded data
        
    Returns:
        pd.DataFrame: DataFrame containing the stock data
    """

    print(f"from {ticker} data from {start_date} to {end_date}")

    data = yf.download(ticker,start = start_date,end = end_date, interval = interval,auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}. Please check the ticker symbol and date range.")
    
    # Save data if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path)
        print(f"Data saved to {save_path}")
    
    return data

def load_stock_data(file_path):
     """
    Load stock data from a CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the stock data
    """
     
     data= pd.read_csv(file_path,index_col=0,parse_dates=True)
     return data
