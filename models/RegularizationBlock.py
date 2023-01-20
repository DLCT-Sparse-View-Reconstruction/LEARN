import torch
from torch import nn

class RegularizationBlock(nn.Module):
    def __init__(self, in_channels=1, filters=48, n_conv=1):
        super(RegularizationBlock, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        self.hid_conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=5, padding=2),
                nn.ReLU()
            ) for i in range(n_conv)])
        
        self.out_conv = nn.Conv2d(in_channels=filters, out_channels=in_channels, kernel_size=5, padding=2)  

        self.apply(self.__init_weights__)

    def __init_weights__(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x_t):
        x = self.init_conv(x_t)
        
        for h_conv in self.hid_conv:
            x = h_conv(x)
        
        x_out = self.out_conv(x)

        return torch.squeeze(x_out)
