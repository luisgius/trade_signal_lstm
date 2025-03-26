import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset 

class FinancialDataset(Dataset):
    def __init__(self, data, seq_length, feature_cols=None, target_col="target", normalize=True):
        """
        Args:
            data (pd.DataFrame): DataFrame containing features and target
            seq_length (int): Length of the sequence for LSTM input
            feature_cols (list): List of column names to use as features
            target_col (str): Column name for the target variable
            normalize (bool): Whether to normalize the features
        """
         
        self.seq_length = seq_length

        # Select features and target
        if feature_cols is None:
              # Use all columns except target as features
              feature_cols = [col for col in data.columns if col != target_col]


        features = data[feature_cols].values
        targets = data[target_col].values

        #Normalize Features
        if normalize:

            #Calculate mean and std
            self.feature_mean = features.mean(axis=0)
            self.feature_std =  features.std(axis=0) 

            # Avoid division by 0
            self.feature_std = np.where(self.feature_std == 0, 1, self.feature_std)
            features = (features - self.feature_mean) / self.feature_std

        self.data = features
        self.targets = targets

    def __len__(self):
         return len(self.data) - self.seq_length
    
    def __getitem__(self,idx):

         # Get a sequence of data
        X = self.data[idx:idx+self.seq_length]
        # Get the target (next day's price)
        y = self.targets[idx+self.seq_length-1] # -1 because we want the target aligned with the last day in sequence
        
        return torch.FloatTensor(X), torch.FloatTensor([y])




