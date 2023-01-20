import torch
from torch import nn

from util.utils import *

from models.RegularizationBlock import RegularizationBlock

class LEARN_RNNCell(nn.Module):
    def __init__(self):
        super(LEARN_RNNCell, self).__init__()
        
        self.reg_x = RegularizationBlock()
        self.reg_y = RegularizationBlock()
        
        self.lamb_t = nn.Parameter(torch.tensor(0.))
    
    def forward(self, d, x_t, fp_operator, fbp_operator):        
        d_t = fp_operator(x_t)
        d_t_reg = self.reg_y(d_t)
        
        d_dif = (d_t_reg + d) - d_t
        
        bp_d_dif = fbp_operator(d_dif)
        
        x_reg = self.reg_x(x_t)
        
        next_x_t = x_t + (self.lamb_t * bp_d_dif) - x_reg
        
        return next_x_t
