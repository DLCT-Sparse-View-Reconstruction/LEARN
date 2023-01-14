import torch
from torch import nn

from util.utils import *

from models.LEARN_RNNCell import LEARN_RNNCell

class LEARN(nn.Module):
    def __init__(self, fp_operator, fbp_operator, n_its):
        super(LEARN, self).__init__()
        self.fp_operator = fp_operator
        self.fbp_operator = fbp_operator
        
        self.learn_rnn_cells = nn.ModuleList([LEARN_RNNCell() for _ in range(n_its)])
    
    def forward(self, d):
        x_t = normlize_tensor(self.fbp_operator(d)).type_as(d)[None, ...]

        for learn_cell in self.learn_rnn_cells:
            x_t = learn_cell(d, x_t, self.fp_operator, self.fbp_operator)
        
        return x_t
