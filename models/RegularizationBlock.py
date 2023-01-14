import torch
from torch import nn

class RegularizationBlock(nn.Module):
    def __init__(self, in_channels=1, filters=48, n_conv=3):
        super(RegularizationBlock, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        self.hid_conv = nn.ModuleList([nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=5, padding=2) 
                                       for i in range(n_conv)])
        
        self.out_conv = nn.Conv2d(in_channels=filters, out_channels=in_channels, kernel_size=5, padding=2)  
    
    def forward(self, x_t):
        x = self.init_conv(x_t)
        
        for h_conv in self.hid_conv:
            x = h_conv(x)
        
        x_out = self.out_conv(x)

        return torch.squeeze(x_out)