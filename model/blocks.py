import torch.nn as nn

class BasicDiscBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 2, padding = 1):
        super(BasicDiscBlock,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True))
    
    def forward(self, input):
        output  = self.main(input)
        return output


class BasicGenBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 2, padding = 1):
        super(BasicGenBlock,self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
    
    def forward(self, input):
        output  = self.main(input)
        return output