import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from utils.sequence_dataset import FinancialDataset  # Import your dataset class
from models.lstm_model import FinancialLSTM  # Import your model class
import torch.nn as nn
import torch.optim as optim

def prepare_financial_data(filepath,  seq_length=20, train_split=0.7, val_split=0.15):

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    
    # Define feature and target columns
    feature_cols = [col for col in df.columns if col != 'target']
    target_col = 'target'
    
    # Calculate split points
    train_size = int(len(df) * train_split)
    val_size = int(len(df) * val_split)
    
    # Split data
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size+val_size]
    test_data = df.iloc[train_size+val_size:]
    
    # Create datasets
    train_dataset = FinancialDataset(train_data, seq_length, feature_cols, target_col)
    val_dataset = FinancialDataset(val_data, seq_length, feature_cols, target_col)
    test_dataset = FinancialDataset(test_data, seq_length, feature_cols, target_col)
    
    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, feature_cols


def train_lstm_model(model, train_loader, val_loader, test_loader, epochs=50, learning_rate=0.001):

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    #Loss function and optimizer
    criterion= nn.MSELoss()
    optimizer= optim.Adam(model.parameters(), lr=learning_rate)

    # For tracking best model
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0

    #training loop
    for epoch in range(epochs):
        model.train()
        train_loss=0
        for X_batch, y_batch in train_loader:
            # Move data to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:  # Use val_loader, not test_loader
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    
    return model
