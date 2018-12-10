import torch
import torch.nn as nn
import numpy as np

from .blocks import ResBlockGen, ResBlockDes, NonLocalBlock
from .sync_batchnorm import DataParallelWithCallback, SynchronizedBatchNorm2d
from .spectral import spectral_norm, SpectralNorm

def get_biggan(num_class, gpus = []):
    if len(gpus) > 0:
        device = torch.device('cuda:' + str(gpus[0]))
        generator = Generator(num_class=num_class).to(device)
        discriminator = Discriminator(num_class=num_class).to(device)

        if len(gpus) > 1:
            generator = nn.DataParallel(generator, device_ids=gpus)
            discriminator = nn.DataParallel(discriminator, device_ids=gpus)
    else:
        generator = Generator(num_class=num_class)
        discriminator = Discriminator(num_class=num_class)
    
    return discriminator, generator
        

class Generator(nn.Module):
    def __init__(self, noise_size = 120, num_class = 0, ch = 64):
        super(Generator,self).__init__()
        self.noise_size = noise_size
        self.num_classes = num_class
        self.ch = ch
        self.num_blocks = 6
        self.linear = SpectralNorm(nn.Linear(num_class, 128, bias=False))
        self.dense = SpectralNorm(nn.Linear(int(self.noise_size/self.num_blocks), 4 * 4 * 16 * ch))

        self.blocks = nn.ModuleList([
            ResBlockGen(16 * ch, 16 * ch, num_class),
            ResBlockGen(16 * ch, 8 * ch, num_class),
            ResBlockGen(8 * ch, 4 * ch, num_class),
            ResBlockGen(4 * ch, 2 * ch, num_class),
            NonLocalBlock(2 * ch),
            ResBlockGen(2 * ch, 1 * ch, num_class)
        ])
        self.sync_bn = SynchronizedBatchNorm2d(1 * ch)
        self.last = nn.Sequential(
            nn.ReLU(True),
            SpectralNorm(nn.Conv2d(1*ch, 3, 3, padding=1)),
            nn.Tanh()
        )
    
    def forward(self, input, class_id):
        codes = torch.split(input, int(self.noise_size/self.num_blocks), 1)
        embeddings = self.linear(class_id)

        out = self.dense(codes[0])
        out = out.view(-1, 16* self.ch, 4, 4)
        c_id = 1
        for layer in self.blocks:
            if isinstance(layer, ResBlockGen):
                condition = torch.cat([codes[c_id], embeddings], 1)
                out = layer(out, condition)
                c_id += 1
            else:
                out = layer(out)
        out = self.sync_bn(out)
        out = self.last(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, num_class = 0, ch = 64):
        super(Discriminator,self).__init__()
        self.num_classes = num_class
        self.ch = ch
        
        self.first_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 1*ch, 3, padding=1)),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(1*ch, 1*ch, 3, padding=1)),
            nn.AvgPool2d(2)
        )

        self.first_skip = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(nn.Conv2d(3, 1*ch, 1)),
        )

        self.main = nn.Sequential(
            ResBlockDes(1*ch, 1*ch, stride = 2),
            NonLocalBlock(1*ch),
            ResBlockDes(1*ch, 2*ch, stride = 2),
            ResBlockDes(2*ch, 4*ch, stride = 2),
            ResBlockDes(4*ch, 8*ch, stride = 2),
            ResBlockDes(8*ch, 16*ch, stride = 2),
            ResBlockDes(16*ch, 16*ch, stride = 1),
            nn.ReLU(True)
        )

        self.embedding = nn.Embedding(num_class, 16*ch)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.embedding = spectral_norm(self.embedding)
        self.linear = SpectralNorm(nn.Linear(16*ch, 1))
    
    def forward(self, input, class_id):
        out = self.first_conv(input)
        out = out + self.first_skip(input)

        out = self.main(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        linear_out = self.linear(out).squeeze(1)
        
        return linear_out + (out * self.embedding(class_id)).sum(1)