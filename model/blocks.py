import torch.nn as nn
import numpy as np

class BasicDiscBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 2, padding = 1):
        super(BasicDiscBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 4, stride , padding, bias = False)
        self.relu = nn.LeakyReLU(0.2,inplace= True)
        self.bn1 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out  = self.conv1(x)
        out = self.relu(out)
        out =self.bn1(out)
        return out


class BasicGenBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 2, padding = 1):
        super(BasicGenBlock,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride, padding, bias = False)
        self.relu = nn.ReLU(inplace= True)
        self.bn1 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out  = self.conv1(x)
        out =self.bn1(out)
        out = self.relu(out)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator_DCGAN(nn.Module):
    def __init__(self, input_size, in_ch):
        super(Discriminator_DCGAN,self).__init__()

        if input_size and input_size & (input_size - 1) == 0:
            self.input_size = input_size
        else:
            raise ValueError('Input size is not a power of 2!')
            
        self.in_ch = in_ch  
        self.discriminator = self._make_discriminator()

    def _make_discriminator(self):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(self.in_ch,self.input_size,4,2,1,bias = False),
                nn.LeakyReLU(0.2, inplace = True)))

        cur_size = int(self.input_size/2)
        features = self.input_size
        while cur_size > 4:
            layers.append(BasicDiscBlock(features,features * 2))
            features*= 2
            cur_size/=2
        
        layers.append(
            nn.Sequential(
                nn.Conv2d(features,1,4,1,0, bias =False),
                nn.Sigmoid()))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.discriminator(x)
        return output

class Generator_DCGAN(nn.Module):
    def __init__(self, output_size, in_ch = 100, out_ch = 3):
        super(Generator_DCGAN,self).__init__()
        self.output_size = output_size
        self.in_ch = in_ch
        self.out_ch  = out_ch
        self.generator = self._make_generator()
    
    def _make_generator(self):
        layers = []
        features = (2 ** (int(np.log2(self.output_size)/2))) * self.output_size
        layers.append(BasicGenBlock(self.in_ch, features, 1, 0))
        out_size = 4
        while out_size < int(self.output_size/2):
            layers.append(BasicGenBlock(features, int(features/2)))
            features = int(features/2)
            out_size*= 2
        
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(features, self.out_ch, 4, 2, 1, bias =False),
                nn.Tanh()))

        return nn.Sequential(*layers) 

    def forward(self, x):
        output = self.generator(x)
        return output