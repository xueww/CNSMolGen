import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDirLSTM(nn.Module):

    def __init__(self, input_dim=110, hidden_dim=256, layers=2):
        super(BiDirLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, dropout=0.3, bidirectional=True)
        self.norm_0 = nn.LayerNorm(input_dim, eps=.001)
        self.norm_1 = nn.LayerNorm(2 * hidden_dim, eps=.001)
        self.wpred = nn.Linear(2 * hidden_dim, input_dim)
        self._init_weights()

    def _init_weights(self):
        # Initialize LSTM weights
        for name, param in self.blstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                # Set the forget gate bias to 1
                nn.init.constant_(param.data[self.hidden_dim:2*self.hidden_dim], 1)
        
        # Initialize weights and biases for linear layer
        nn.init.xavier_uniform_(self.wpred.weight)
        nn.init.constant_(self.wpred.bias, 0)

    def _init_hidden(self, batch_size: int, device: torch.device):
        '''Initialize hidden states'''
        weight = next(self.parameters()).data
        return (weight.new(self.layers * 2, batch_size, self.hidden_dim).zero_().to(device),
                weight.new(self.layers * 2, batch_size, self.hidden_dim).zero_().to(device))

    def new_sequence(self, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        '''Prepare model for a new sequence'''
        self.hidden = self._init_hidden(batch_size, device)

    def forward(self, input: torch.Tensor, next_prediction: str = 'right', device: torch.device = torch.device('cpu')) -> torch.Tensor:
        '''Forward computation'''
        if next_prediction == 'left':
            input = torch.flip(input, [0])

        norm_0 = self.norm_0(input)
        out, self.hidden = self.blstm(norm_0, self.hidden)
        for_out = out[-1, :, :self.hidden_dim]
        back_out = out[0, :, self.hidden_dim:]
        bmerge = torch.cat((for_out, back_out), -1)
        norm_1 = self.norm_1(bmerge)
        pred = self.wpred(norm_1)
        return pred