import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FinancialLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(FinancialLSTM,self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #LSTM Model
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout)
        
        #Fully connected Output layer 
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        # x shape: (batch_size, sequence_length, input_size)

        device = x.device

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward Propagation
        out,_ = self.lstm(x,[h0,c0])

        # Decode hidden state of last step
        out = self.fc(out[:,-1,:])

        return out
