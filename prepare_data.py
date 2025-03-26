# prepare_data.py
import os
import argparse
import pandas as pd
from utils.data_loader import data_acquisition
from utils.technical_indicators import add_technical_indicators

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download and prepare stock data')
    parser.add_argument('--ticker', type=str, default='GOOGL', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2017-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-01-01', help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    #create directories if not exist
    os.makedirs("data/raw",exist_ok=True)
    os.makedirs("data/processed",exist_ok=True)

    #Download raw data
    raw_data_path = f"data/raw/{args.ticker}_{args.start_date}_{args.end_date}.csv"
    df = data_acquisition(args.ticker, args.start_date, args.end_date, save_path=raw_data_path)

    #add techincal indicators
    df_indicators = add_technical_indicators(df)

    #Save processed data
    processed_data_path = f"data/processed/{args.ticker}_{args.start_date}_{args.end_date}_processed.csv"
    df_indicators.to_csv(processed_data_path)

    print(f"Data processing complete. Processed data saved to {processed_data_path}")
    print(f"Raw data shape: {df.shape}")
    print(f"Processed data shape: {df_indicators.shape}")


if __name__== "__main__":
    main()


"""
To run this script, you would use: 

python prepare_data.py --ticker GOOGL --start_date 2018-01-01 --end_date 2023-01-01

"""
