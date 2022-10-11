import torch
from torch import nn
from torch.nn import functional as F




# Code adapted from https://github.com/google-research/google-research/tree/master/aloe/aloe
class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

# Code adapted from https://github.com/google-research/google-research/tree/master/aloe/aloe
class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# Code adapted from https://github.com/google-research/google-research/tree/master/aloe/aloe  
NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}


class EBM_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=-1, bn=False, activation='elu'):
        super(EBM_MLP, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.bn = bn
        
        hidden_dims = [input_dim] + [hidden_dim] * 3 + [1]
        
        layers_list = []
        
        for k in range(len(hidden_dims)-1):
            layers_list.append(nn.Linear(hidden_dims[k], hidden_dims[k+1]))
            if k!=len(hidden_dims)-2:
                if self.bn:
                    layers_list.append(nn.BatchNorm1d(hidden_dims[k+1]))
                layers_list.append(NONLINEARITIES[self.activation])
                if self.dropout > 0:
                    layers_list.append(nn.Dropout(self.dropout))

        
        self.whole_model = nn.Sequential(*layers_list)
      
    def forward(self, x):
        energy = self.whole_model(x)      
        return energy