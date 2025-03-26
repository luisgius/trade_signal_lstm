import argparse
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Import your custom modules
from utils.data_loader import data_acquisition
from utils.technical_indicators import add_technical_indicators
from utils.sequence_dataset import FinancialDataset
from models.lstm_model import FinancialLSTM
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from predict import load_model, make_predictions, evaluate_predictions, plot_predictions
from backtest import generate_signals, backtest_strategy, calculate_performance_metrics, plot_strategy_performance
from train import train_lstm_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run complete trading system pipeline")
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'backtest', 'full'], 
                      default='full', help="Operation mode")
    parser.add_argument('--ticker', type=str, default='GOOGL', help="Stock ticker symbol")
    parser.add_argument('--start_date', type=str, default='2018-01-01', help="Start date for data")
    parser.add_argument('--end_date', type=str, default='2023-01-01', help="End date for data")
    parser.add_argument('--model_path', type=str, default='models/saved/lstm_model.pth', 
                      help="Path to save/load model")
    parser.add_argument('--threshold', type=float, default=0.001, 
                      help="Threshold for trading signals")
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Step 1: Data preparation
    if args.mode in ['train', 'full']:
        logger.info(f"Downloading and processing data for {args.ticker}")
        try:
            # Download data
            raw_data_path = f'data/raw/{args.ticker}_{args.start_date}_{args.end_date}.csv'
            data = data_acquisition(args.ticker, args.start_date, args.end_date, save_path=raw_data_path)
            
            # Add technical indicators
            processed_data = add_technical_indicators(data)
            processed_data_path = f'data/processed/{args.ticker}_{args.start_date}_{args.end_date}_processed.csv'
            processed_data.to_csv(processed_data_path)
            logger.info(f"Data processed and saved to {processed_data_path}")
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            return
    
    # Step 2: Load processed data
    try:
        processed_data_path = f'data/processed/{args.ticker}_{args.start_date}_{args.end_date}_processed.csv'
        df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded processed data with shape {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load processed data: {str(e)}")
        return
    
    # Define feature columns and target
    feature_cols = [col for col in df.columns if col != 'target']
    target_col = 'target'
    
    # Step 3: Prepare datasets
    seq_length = 20
    dataset = FinancialDataset(df, seq_length=seq_length, 
                              feature_cols=feature_cols, target_col=target_col)
    
    # Split data
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Step 4: Model training (if in train mode)
    if args.mode in ['train', 'full']:
        logger.info("Starting model training")
        try:
            # Initialize model
            model = FinancialLSTM(
                input_size=len(feature_cols),
                hidden_size=128,
                num_layers=2,
                output_size=1,
                dropout=0.2
            )
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train model
            # (Call your train_lstm_model function here)
            model = train_lstm_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=50,
            learning_rate=0.001
            )
            
            # Save model parameters for later use
            model_info = {
                'state_dict': model.state_dict(),
                'params': {
                    'input_size': len(feature_cols),
                    'hidden_size': 128,
                    'num_layers': 2,
                    'output_size': 1,
                    'dropout': 0.2
                },
                'device_mapping': str(next(model.parameters()).device)  # Critical for reloading
            }
            save_dir = os.path.dirname(args.model_path)
            print(f"Attempting to save model to directory: {save_dir}")
            print(f"This directory exists: {os.path.exists(save_dir)}")
            
            
            #torch.save(model.state_dict(), args.model_path)
            torch.save(model_info, args.model_path)
            logger.info(f"Model trained and saved to {args.model_path}")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            if args.mode == 'train':
                return
    
    # Step 5: Prediction (if in predict or backtest mode)
    if args.mode in ['predict', 'backtest', 'full']:
        logger.info("Making predictions with trained model")
        try:
            # Load model
            model_info = torch.load(args.model_path)
            model_params = model_info['params']
            model = FinancialLSTM(**model_params)
            model.load_state_dict(model_info['state_dict'])
            
            # Make predictions
            predictions, actuals = make_predictions(model, test_loader)
            
            # Evaluate predictions
            metrics = evaluate_predictions(actuals, predictions)
            logger.info("Model Performance Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value}")
            
            # Plot predictions (select a subset for clarity)
            plot_range = min(100, len(predictions))
            # Extract dates from the test portion of the DataFrame
            test_dates = df.index[-(test_size+seq_length):-seq_length][-plot_range:]
            plot_predictions(
                test_dates[-plot_range:], 
                actuals[-plot_range:], 
                predictions[-plot_range:],
                title=f"{args.ticker} Return Predictions"
            )
            logger.info(f"Prediction plot saved")
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            if args.mode == 'predict':
                return
    
    # Step 6: Backtesting (if in backtest mode)
    if args.mode in ['backtest', 'full']:
        logger.info("Backtesting trading strategy")
        try:
            # Generate trading signals
            signals = generate_signals(predictions, threshold=args.threshold)
            
            # Prepare price data for backtesting
            test_dates = df.index[-(test_size+seq_length):-seq_length]
            if len(test_dates) != len(predictions):
                test_dates = test_dates[:len(predictions)]
            
            # Get corresponding price data
            price_data = df.loc[test_dates, 'close'].values
            
            # Create DataFrame for backtesting
            backtest_df = pd.DataFrame({
                'close': price_data,
                'signal': signals.flatten()
            }, index=test_dates)
            
            # Run backtest
            results = backtest_strategy(backtest_df['signal'], backtest_df['close'])
            
            # Calculate performance metrics
            performance = calculate_performance_metrics(results)
            logger.info("Trading Strategy Performance:")
            for metric, value in performance.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Save results
            results.to_csv(f'results/{args.ticker}_backtest_results.csv')
            
            # Plot strategy performance
            # Plot strategy performance
            plot_strategy_performance(
                results, 
                title=f"{args.ticker} Trading Strategy Performance",
                initial_capital=10000
            )
            
            logger.info(f"Backtest results saved")
        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
    
    logger.info("Trading system execution completed")

if __name__ == "__main__":
    main()