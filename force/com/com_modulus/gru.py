import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):

    """  
    A single ConvGRU cell for processing spatiotemporal data.  
    """  
    def __init__(self, input_dim=192+128, hidden_dim=128, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=1)

    def forward(self,x, h):

        if h is None:  
            h = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3)).to(x.device) 

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class ConvGRU(nn.Module):  
    """  
    ConvGRU module for processing sequences of images.  
    """  
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1):  
        super(ConvGRU, self).__init__()  
        self.num_layers = num_layers  
        self.hidden_dim = hidden_dim  
        self.layers = nn.ModuleList(  
            [ConvGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size) for i in range(num_layers)]  
        )  

    def forward(self, x, hidden_states=None):  
        """  
        Args:  
            x: Input tensor of shape (S, B, C, H, W) - sequence of images  
            hidden_states: Initial hidden states, list of tensors (optional)  
        Returns:  
            outputs: Hidden states for all time steps (S, B, C_hidden, H, W)  
            last_states: Final hidden states for each layer  
        """  
        seq_len, batch_size, channels, height, width = x.size()  

        # Initialize hidden states if not provided  
        if hidden_states is None:  
            hidden_states = [None] * self.num_layers  

        current_input = x  
        last_states = []  
        outputs = []  

        for layer_idx, layer in enumerate(self.layers):  
            # Process the input sequence through the ConvGRU layer  
            layer_hidden = hidden_states[layer_idx]  
            layer_outputs = []  

            for t in range(seq_len):  
                layer_hidden = layer(current_input[t], layer_hidden)  
                layer_outputs.append(layer_hidden)  

            # Stack the output of all time steps for the current layer  
            layer_outputs = torch.stack(layer_outputs, dim=0)  # Shape: (S, B, C_hidden, H, W)  
            current_input = layer_outputs  
            outputs.append(layer_outputs)  
            last_states.append(layer_hidden)  

        outputs = torch.stack(outputs, dim=0)  # Shape: (num_layers, S, B, C_hidden, H, W)  
        return outputs[-1], last_states  # Return the final layer output and final hidden states  