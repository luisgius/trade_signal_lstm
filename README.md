# trade_signal_lstm

IMPORTANT: This Project is on progress, results probably are not accurate and needs improvements on trading strategies.

## Overview
This project implements a deep learning-based trading signal model using LSTM networks to predict stock price movements. The system downloads financial data, calculates technical indicators, trains an LSTM model, and backtests a trading strategy based on the model's predictions.

## Features
- Data acquisition from Yahoo Finance
- Technical indicator calculation
- LSTM model for time series prediction
- Trading signal generation
- Strategy backtesting and performance evaluation
- Comprehensive visualization

## Project Structure
dl_trading_signal_project/
├── data/
│   ├── raw/                  # Raw downloaded data
│   └── processed/            # Processed data with features
├── models/
│   ├── lstm_model.py         # LSTM model architecture
│   └── saved/                # Saved model weights
├── utils/
│   ├── data_loader.py        # Functions for data loading
│   ├── technical_indicators.py # Functions for technical indicators
│   └── sequence_dataset.py   # Dataset class for LSTM input
├── results/
│   └── plots/                # Performance visualizations
├── train.py                  # Model training script
├── predict.py                # Prediction functions
├── backtest.py               # Trading strategy and backtesting
└── run_trading_system.py     # Main integration script

## Results
The system evaluates model performance using both statistical metrics (MSE, RMSE, Directional Accuracy) and financial metrics (Returns, Sharpe Ratio, Max Drawdown). Results are saved in the results/ directory.

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/dl-trading-signal-project.git
cd dl-trading-signal-project

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline (data download, processing, training, backtesting)
python run_trading_system.py --mode full --ticker GOOGL --start_date 2018-01-01 --end_date 2023-01-01

# Only train the model
python run_trading_system.py --mode train --ticker GOOGL

# Only run prediction and evaluation
python run_trading_system.py --mode predict --model_path models/saved/lstm_model.pth

# Only run backtesting
python run_trading_system.py --mode backtest --model_path models/saved/lstm_model.pth --threshold 0.002


