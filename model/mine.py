import torch
import torch.nn as nn

class StatisticNetworkDCGAN(nn.Module):
    
    def __init__(self, input_size, in_ch, noise_size = 120):
        super(StatisticNetworkDCGAN, self).__init__()
        if input_size and input_size & (input_size - 1) == 0:
            self.input_size = input_size
        else:
            raise ValueError('Input size is not a power of 2!')
        
        self.noise_size = noise_size
        self.input_size = input_size
        self.in_ch = in_ch

        self.conv = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.batch_norm = nn.ModuleList()

        self.__build_network()

    def __build_network(self):
        
        self.conv0 = nn.Conv2d(self.in_ch, self.input_size, 4, 2, 1, bias = False)
        self.linear0 = nn.Linear(self.noise_size, self.input_size)
        
        cur_size = int(self.input_size/2)
        features = self.input_size
        while cur_size > 4:
            add_c = nn.Conv2d(features,features * 2, 4, 2, 1,bias = False)
            self.conv.append(add_c)
            self.batch_norm.append(nn.BatchNorm2d(features * 2))
            add_l = nn.Linear(self.noise_size, features * 2)
            self.linear.append(add_l)
            features*= 2
            cur_size/=2
        
        self.linear_last = nn.Linear(cur_size * cur_size * features, 1)
        
    def forward(self, x, z):
        output_1 = self.conv0(x)
        output_2 = self.linear0(z)

        output = output_1 + output_2
        output = nn.functional.elu(output)

        for i in range(0, len(self.conv)):
            output_1 = self.conv[i](output)
            output_2 = self.linear[i](z)
            output = output_1 + output_2
            output = self.batch_norm[i](output)
            output = nn.functional.elu(output)
            
        output = output.view(output.size(0),1)
        output = self.linear_last(output)

        return output


