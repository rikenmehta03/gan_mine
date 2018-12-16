from .blocks import BasicDiscBlock, BasicGenBlock
from .spectral import SpectralNorm, spectral_norm
import torch
import torch.nn as nn
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_dcgan(image_size,in_ch,sn=False,device = torch.device('cpu')):
    discriminator = Discriminator_DCGAN(image_size, in_ch,sn).to(device)
    generator = Generator_DCGAN(image_size,sn=sn).to(device)
    discriminator.apply(weights_init)
    generator.apply(weights_init)
    return discriminator, generator

class Discriminator_DCGAN(nn.Module):
    def __init__(self, input_size, in_ch,sn=False):
        super(Discriminator_DCGAN,self).__init__()

        if input_size and input_size & (input_size - 1) == 0:
            self.input_size = input_size
        else:
            raise ValueError('Input size is not a power of 2!')
        self.sn = sn    
        self.in_ch = in_ch  
        self.discriminator = self._make_discriminator()

    def _make_discriminator(self):
        layers = []
        conv = nn.Conv2d(self.in_ch,self.input_size,4,2,1,bias = False)
        if self.sn:
            conv = spectral_norm(conv)    

        layers.append(
            nn.Sequential(
                conv,
                nn.LeakyReLU(0.2, inplace = True)))

        cur_size = int(self.input_size/2)
        features = self.input_size
        while cur_size > 4:
            layers.append(BasicDiscBlock(features,features * 2, self.sn))
            features*= 2
            cur_size/=2
        
        conv = nn.Conv2d(features, 1,4,1,0, bias=False)
        if self.sn:
            conv = spectral_norm(conv)

        layers.append(
            nn.Sequential(
                conv,
                nn.Sigmoid()))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.discriminator(x)
        return output.view(-1, 1).squeeze(1)

class Generator_DCGAN(nn.Module):
    def __init__(self, output_size, in_noise = 100, sn=False, out_ch = 3):
        super(Generator_DCGAN,self).__init__()
        self.output_size = output_size
        self.sn = sn
        self.in_noise = in_noise
        self.out_ch  = out_ch
        self.generator = self._make_generator()
    
    def _make_generator(self):
        layers = []
        features = (2 ** (int(np.log2(self.output_size)/2))) * self.output_size
        layers.append(BasicGenBlock(self.in_noise, features, self.sn, 1, 0))
        out_size = 4
        while out_size < int(self.output_size/2):
            layers.append(BasicGenBlock(features, int(features/2),self.sn))
            features = int(features/2)
            out_size*= 2
        
        conv = nn.ConvTranspose2d(features, self.out_ch, 4, 2, 1, bias =False)

        if self.sn:
            conv = spectral_norm(conv)
            
        layers.append(
            nn.Sequential(
                conv,
                nn.Tanh()))
        
        return nn.Sequential(*layers) 

    def forward(self, x):
        output = self.generator(x)
        return output