# Imports for prediction script
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.lstm_model import FinancialLSTM
from utils.sequence_dataset import FinancialDataset
from torch.utils.data import DataLoader
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

def load_model(model_path, model_params):
    """
    Load a trained LSTM model from disk
    
    Args:
        model_path (str): Path to the saved model file
        model_params (dict): Dictionary containing model parameters
        
    Returns:
        model: Loaded PyTorch model
    """
    # Function implementation - load model, set to eval mode, etc.
    # Initialize model architecture
    model = FinancialLSTM(**model_params)
    
    # Load trained weights (with cross-device compatibility)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))
    )
    
    # Set model to evaluation mode
    model.eval()
    
    return model  # ← Returns loaded model ready for inference


def make_predictions(model, test_loader, device="cpu"):
    """
    Generate predictions using the trained model
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader containing test data
        device: Device to run predictions on (CPU/GPU)
        
    Returns:
        tuple: Predicted values and actual values
    """
    # Function implementation - run model on test data, collect predictions
    model.to(device)
    model.eval()

    predictions=[]
    true_targets=[]

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Move data to the correct device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            # Store results (convert to CPU and then NumPy)
            predictions.append(outputs.cpu().numpy())
            true_targets.append(y_batch.cpu().numpy())

    # Concatenate all batches
    return np.concatenate(predictions), np.concatenate(true_targets)


def evaluate_predictions(y_true, y_pred):
    """
    Calculate performance metrics for model predictions
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        dict: Dictionary containing various performance metrics
    """
    # Ensure arrays are flattened
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate traditional regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate directional accuracy (how often the sign is correct)
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    # Calculate correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Return metrics dictionary
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Directional Accuracy': directional_accuracy,
        'Correlation': correlation
    }
    
    return metrics




def plot_predictions(dates, y_true, y_pred, title="Model Predictions vs Actual",show=True):
    """
    Create institutional-quality visualization of predictions vs actual values
    
    Args:
        dates: Array of dates for x-axis (datetime objects)
        y_true: Array of true values
        y_pred: Array of predicted values
        title: Plot title
    
    Saves:
        High-resolution PNG of the visualization
    """
    plt.style.use('seaborn-darkgrid')  # Professional style
    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
    
    # Error checking
    if len(dates) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("All input arrays must have the same length")

    # Main price plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, y_true, color='#2ca02c', lw=1.5, label='Actual', alpha=0.9)
    ax1.plot(dates, y_pred, color='#d62728', lw=1.2, ls='--', label='Predicted', alpha=0.9)
    
    # Professional formatting
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.2f}'))
    ax1.set_ylabel('Price', fontsize=12)
    
    # Prediction error fill
    ax1.fill_between(dates, y_true, y_pred, 
                    where=(y_pred > y_true), 
                    facecolor='#ff7f0e', alpha=0.15, interpolate=True)
    
    # Residual plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    residuals = y_pred - y_true
    ax2.bar(dates, residuals, color=np.where(residuals>=0, '#2ca02c', '#d62728'), width=1, alpha=0.6)
    
    # Annotations
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    error_text = f'MSE: ${mse:,.2f}\nMAE: ${mae:,.2f}\nCorr: {np.corrcoef(y_true, y_pred)[0,1]:.2%}'
    ax1.text(0.02, 0.95, error_text, transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.85))
    
    # Final touches
    plt.suptitle(title, y=0.95, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save unless only showing
    if not show:
        plt.savefig(f"{title.replace(' ', '_')}_analysis.png", 
                   dpi=300, 
                   bbox_inches='tight',
                   transparent=False)
    
    # Display control
    if show:
        plt.show()
    else:
        plt.close()
    