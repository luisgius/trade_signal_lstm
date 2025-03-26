import argparse
from predict import load_model, make_predictions, evaluate_predictions
from train import prepare_financial_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import sys
import pandas as pd
import numpy as np

def generate_signals(predictions, threshold=0.001):
    """
    Generate trading signals based on predicted returns
    
    Args:
        predictions (array): Array of predicted returns
        threshold (float): Minimum return threshold to generate a signal
        
    Returns:
        array: Array of signals (1=buy, -1=sell, 0=hold)
    """
    signals = np.zeros_like(predictions)
    
    # Buy signal when predicted return is above threshold
    signals[predictions > threshold] = 1
    
    # Sell signal when predicted return is below negative threshold
    signals[predictions < -threshold] = -1
    
    return signals

def backtest_strategy(signals, prices, initial_capital=10000.0, commission=0.001):
    """
    Backtest a trading strategy based on signals
    
    Args:
        signals (array): Array of trading signals (1=buy, -1=sell, 0=hold)
        prices (array): Array of asset prices
        initial_capital (float): Starting capital for the strategy
        commission (float): Trading commission as a percentage
        
    Returns:
        pd.DataFrame: DataFrame with strategy performance metrics
    """
    # Create a DataFrame with dates, prices and signals
    backtest_df = pd.DataFrame({
        'price': prices,
        'signal': signals
    }, index=prices.index)
    
    # Create position column (1 = long, -1 = short, 0 = cash)
    backtest_df['position'] = backtest_df['signal']
    
    # Calculate returns based on the price
    backtest_df['market_return'] = backtest_df['price'].pct_change()
    
    # Calculate strategy returns
    # Strategy return is the return based on previous position
    backtest_df['strategy_return'] = backtest_df['market_return'] * backtest_df['position'].shift(1)
    
    # Handle commission costs
    # Identify when position changes
    backtest_df['position_change'] = backtest_df['position'].diff().fillna(0).abs()
    # Apply commission when position changes
    backtest_df['commission_cost'] = backtest_df['position_change'] * commission
    
    # Adjust strategy returns for commission costs
    backtest_df['strategy_return_after_cost'] = backtest_df['strategy_return'] - backtest_df['commission_cost']
    
    # Calculate cumulative returns
    backtest_df['cumulative_market_return'] = (1 + backtest_df['market_return']).cumprod() - 1
    backtest_df['cumulative_strategy_return'] = (1 + backtest_df['strategy_return_after_cost']).cumprod() - 1
    
    # Calculate strategy value
    backtest_df['strategy_value'] = initial_capital * (1 + backtest_df['cumulative_strategy_return'])
    
    return backtest_df

def calculate_performance_metrics(backtest_df, risk_free_rate=0.02/252):
    """
    Calculate performance metrics for a trading strategy
    
    Args:
        backtest_df (pd.DataFrame): DataFrame with strategy returns
        risk_free_rate (float): Daily risk-free rate
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Extract strategy returns
    strategy_returns = backtest_df['strategy_return_after_cost'].dropna()
    
    # Calculate annualized return
    annualized_return = (1 + strategy_returns.mean()) ** 252 - 1
    
    # Calculate annualized volatility
    annualized_vol = strategy_returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol
    
    # Calculate maximum drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
    
    # Calculate profit factor
    profit_factor = np.abs(strategy_returns[strategy_returns > 0].sum() / 
                          strategy_returns[strategy_returns < 0].sum())
    
    return {
        'Total Return': backtest_df['cumulative_strategy_return'].iloc[-1],
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor
    }

def plot_strategy_performance(backtest_df, benchmark_returns=None, title="Trading Strategy Performance", initial_capital=10000):
    """
    Create visualization of trading strategy performance
    
    Args:
        backtest_df (pd.DataFrame): DataFrame with backtest results
        benchmark_returns (pd.DataFrame, optional): DataFrame with benchmark returns
        title (str): Plot title
        initial_capital (float): Initial capital for the strategy
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Portfolio value plot
    ax1 = plt.subplot(3, 1, 1)
    backtest_df['strategy_value'].plot(ax=ax1, color='blue', lw=2)
    ax1.set_title('Portfolio Value Over Time', fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True)
    
    # Calculate buy & hold performance for comparison
    if 'close' in backtest_df.columns:
        buy_hold_value = initial_capital * backtest_df['close'] / backtest_df['close'].iloc[0]
        buy_hold_value.plot(ax=ax1, color='gray', linestyle='--', lw=1.5, label='Buy & Hold')
    
    ax1.legend()
    
    # Cumulative returns plot
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    if 'cumulative_strategy_return' in backtest_df.columns:
        backtest_df['cumulative_strategy_return'].plot(ax=ax2, color='green', lw=2, label='Strategy')
    
    if 'cumulative_market_return' in backtest_df.columns:
        backtest_df['cumulative_market_return'].plot(ax=ax2, color='gray', linestyle='--', lw=1.5, label='Market')
    
    ax2.set_title('Cumulative Returns', fontweight='bold')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True)
    ax2.legend()
    
    # Drawdown plot
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    
    # Calculate drawdowns if not already in DataFrame
    if 'drawdown' not in backtest_df.columns and 'strategy_return_after_cost' in backtest_df.columns:
        # Calculate cumulative returns if not already present
        if 'cum_returns' not in backtest_df.columns:
            backtest_df['cum_returns'] = (1 + backtest_df['strategy_return_after_cost']).cumprod()
        
        # Calculate drawdowns
        backtest_df['peak'] = backtest_df['cum_returns'].cummax()
        backtest_df['drawdown'] = (backtest_df['cum_returns'] / backtest_df['peak']) - 1
    
    if 'drawdown' in backtest_df.columns:
        backtest_df['drawdown'].plot(ax=ax3, color='red', lw=1.5)
    
    ax3.set_title('Drawdowns', fontweight='bold')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True)
    ax3.set_ylim(bottom=backtest_df['drawdown'].min() * 1.1 if 'drawdown' in backtest_df.columns else -0.1, top=0.01)
    
    # Calculate performance metrics
    if 'strategy_return_after_cost' in backtest_df.columns:
        returns = backtest_df['strategy_return_after_cost'].dropna()
        
        # Annualized return
        ann_return = (1 + returns.mean()) ** 252 - 1
        
        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        # Max drawdown
        max_dd = backtest_df['drawdown'].min() if 'drawdown' in backtest_df.columns else 0
        
        # Add metrics annotation
        metrics_text = f"Ann. Return: {ann_return:.2%}\n"
        metrics_text += f"Ann. Volatility: {ann_vol:.2%}\n"
        metrics_text += f"Sharpe Ratio: {sharpe:.2f}\n"
        metrics_text += f"Max Drawdown: {max_dd:.2%}"
        
        plt.figtext(0.15, 0.01, metrics_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and title
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()







def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Backtest trading strategy using LSTM predictions")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model")
    parser.add_argument('--data_path', type=str, required=True, help="Path to test data")
    parser.add_argument('--threshold', type=float, default=0.001, help="Threshold for signal generation")
    parser.add_argument('--capital', type=float, default=10000.0, help="Initial capital")
    args = parser.parse_args()
    
    # Load data and model
    # Load test data first
    test_data = pd.read_csv(args.data_path)
    feature_cols = [col for col in test_data.columns if col != 'target']
    input_size = len(feature_cols)
    try:
        model_params = {  # Should match training configuration
            "input_size": input_size,    # Number of features
            "hidden_size": 128,  
            "num_layers": 2,
            "output_size": 1,
            "dropout": 0.2
        }
        model = load_model(args.model_path, model_params)
        print(f"✅ Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        sys.exit(1)

    # Generate predictions

    # Use underscores to ignore unwanted returns
    _, _, test_loader, _ = prepare_financial_data(args.data_path)
    predictions, targets = make_predictions(model,test_loader)

    metrics = evaluate_predictions(targets, predictions)
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Create trading signals
    signals = generate_signals(predictions, threshold=args.threshold)

    # Backtest strategy

    # Note: We need to align predictions with dates
    # This may require some adjustment based on your dataset
    aligned_dates = test_data.index[20:]  # Skip first seq_length rows
    if len(aligned_dates) != len(predictions):
        aligned_dates = aligned_dates[:len(predictions)]
    
    backtest_results = backtest_strategy(
        signals,
        test_data.loc[aligned_dates, 'close'].values,
        initial_capital=args.initial_capital
    )

    # Calculate performance metrics
    performance = calculate_performance_metrics(backtest_results)
    print("\nStrategy Performance:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")

    # Visualize results
    plot_strategy_performance(backtest_results)
    
if __name__ == "__main__":
    main()