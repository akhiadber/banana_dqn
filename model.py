import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 64], dropout_p=0.3):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.input_size = state_size
        self.output_size = action_size
        self.hidden_layers = hidden_layers
        self.dropout_p = dropout_p
        
        self.layers = nn.ModuleList([nn.Linear(self.input_size, self.hidden_layers[0])])
        self.layer_sizes = zip(self.hidden_layers[:-1], self.hidden_layers[1:])
        self.layers.extend(nn.Linear(size_1, size_2) for size_1, size_2 in self.layer_sizes)
        
        self.output = nn.Linear(self.hidden_layers[-1], self.output_size)
        
        self.dropout = nn.Dropout(self.dropout_p)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
            #x = self.dropout(x)
            
        action_values = self.output(x)
        
        return action_values