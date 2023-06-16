import torch
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from torch import nn

class lstm(nn.Module):
    def __init__(self, input, output, hidden, layers):
        super(lstm, self).__init__()
        self.input_size = input
        self.output_size = output
        self.hidden_size = hidden
        self.layers = layers

        #self.encoder = nn.Linear(hidden, output)
        self.rnn_cell = nn.LSTM(input_size = input, hidden_size = hidden, num_layers = layers)#, dropout = 0.01)
        self.decoder = nn.Linear(hidden, output)


    def forward(self, x):
        # x is matrix of inputs
        # each cell gets a full pandas row as input
        output, self.hidden = self.rnn_cell(x)
        
        output = self.decoder(output[-1, :])
        return output
    
class deep_linear(nn.Module):
    def __init__(self, input, output, hidden, layers):
        super(deep_linear, self).__init__()
        self.input_size = input
        self.output_size = output
        self.hidden_size = hidden
        self.layers = layers

        self.lin1 = nn.Linear(layers, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, hidden)
        self.lin4 = nn.Linear(hidden, output)


    def forward(self, x):
        # x is matrix of inputs
        # each cell gets a full pandas row as input
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.15, training=self.training)

        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.15, training=self.training)

        x = self.lin3(x)
        x = x.relu()
        x = F.dropout(x, p=0.15, training=self.training)

        x = self.lin4(x)

        return x
    
class recurrent(nn.Module):
    def __init__(self, input, output, hidden, layers):
        super(recurrent, self).__init__()
        self.input_size = input
        self.output_size = output
        self.hidden_size = hidden
        self.layers = layers

        #self.encoder = nn.Linear(hidden, output)
        self.rnn_cell = nn.RNN(input_size = input, hidden_size = hidden, num_layers = layers)#, dropout = 0.01)
        self.decoder = nn.Linear(hidden, output)


    def forward(self, x):
        # x is matrix of inputs
        # each cell gets a full pandas row as input
        output, self.hidden = self.rnn_cell(x)
        
        output = self.decoder(output[-1, :])
        return output
