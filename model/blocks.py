import torch.nn as nn
import torch.nn.functional as F

class ConditionalBatchNorm(nn.Module):
    def __init__(self, in_channel, n_condition=148):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)

        self.embed = nn.Linear(n_condition, in_channel* 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        return gamma * out + beta

class NonLocalBlock(nn.Module):
    """ Non local block"""
    def __init__(self,in_dim,activation=F.relu):
        super(NonLocalBlock,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

        nn.init.xavier_uniform(self.query_conv.weight)
        nn.init.xavier_uniform(self.key_conv.weight)
        nn.init.xavier_uniform(self.value_conv.weight)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class BasicDiscBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, sn, stride = 2, padding = 1):
        super(BasicDiscBlock,self).__init__()
        if sn:
            conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride, padding, bias = False))
        else:
            conv = nn.Conv2d(in_channels, out_channels, 4, stride, padding, bias = False)
        
        self.main = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True))
    
    def forward(self, input):
        output  = self.main(input)
        return output


class BasicGenBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, sn, stride = 2, padding = 1):
        super(BasicGenBlock,self).__init__()
        if sn:
            conv_t = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, 4, stride, padding, bias = False))
        else:
            conv_t = nn.ConvTranspose2d(in_channels, out_channels, 4, stride, padding, bias = False)
        
        self.main = nn.Sequential(
            conv_t,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
    
    def forward(self, input):
        output  = self.main(input)
        return output

class ResBlockGen(nn.Module):

    def __init__(self, in_channels, out_channels, number_class = 0, stride = 1, padding = 1, sn=True):
        super(ResBlockGen,self).__init__()
        
        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=padding)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=padding)
        nn.init.orthogonal(self.conv0.weight.data, 1.)
        nn.init.orthogonal(self.conv1.weight.data, 1.)

        if sn:
            self.conv0 = nn.utils.spectral_norm(self.conv0)
            self.conv1 = nn.utils.spectral_norm(self.conv1)

        if number_class == 0:
            self.bn0 = nn.BatchNorm2d(in_channels)
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn0 = ConditionalBatchNorm(in_channels)
            self.bn1 = ConditionalBatchNorm(out_channels)

        self.activation = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, input, class_id=None):
        bypass = self.upsample(input)
        if class_id is None:
            out = self.bn0(input)
        else:
            out = self.bn0(input, class_id)
        
        out = self.activation(out)
        out = self.upsample(out)
        out = self.conv0(out)

        if class_id is None:
            out = self.bn1(out)
        else:
            out = self.bn1(out, class_id)

        out = self.activation(out)
        out = self.conv1(out)
        return out + bypass

class ResBlockDes(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1, padding = 1, sn=True):
        super(ResBlockDes,self).__init__()
        
        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=padding)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=padding)
        nn.init.orthogonal(self.conv0.weight.data, 1.)
        nn.init.orthogonal(self.conv1.weight.data, 1.)

        if sn:
            self.conv0 = nn.utils.spectral_norm(self.conv0)
            self.conv1 = nn.utils.spectral_norm(self.conv1)

        self.activation = nn.ReLU(True)
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.orthogonal(self.bypass_conv.weight.data)
            self.bypass = nn.Sequential(
                nn.utils.spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

            self.main = nn.Sequential(
                self.activation,
                self.conv0,
                self.activation,
                self.conv1,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.main = nn.Sequential(
                self.activation,
                self.conv0,
                self.activation,
                self.conv1
            )

    def forward(self, input):
        return self.main(input) + self.bypass(input)


