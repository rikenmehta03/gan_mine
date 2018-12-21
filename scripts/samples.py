import argparse
import os
import glob
import torch
import numpy as np
import sys
from scipy.stats import truncnorm
from torchvision import utils
import torch
from torch.autograd import Variable
import torchvision.utils as vutils


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import get_biggan, get_dcgan

def label_sampel(batch_size, num_classes):
    label = torch.LongTensor(batch_size, 1).random_()%num_classes
    one_hot= torch.zeros(batch_size, num_classes).scatter_(1, label, 1)
    return label.squeeze(1).to(device), one_hot.to(device) 

def truncated_z_sample(model, batch_size, noise_size ,truncation, seed=None):
    state = None
    if model == 'dcgan':
        values = truncnorm.rvs(-2, 2, size=(batch_size, noise_size, 1, 1), random_state = state)
    else:
        values = truncnorm.rvs(-2, 2, size=(batch_size, noise_size), random_state = state)
    return torch.Tensor(truncation * values)

def generate_latent_walk(G, i, img_size ,number, device):
    # Code adapted from https://github.com/Zeleni9/pytorch-wgan/blob/master/models/dcgan.py

    if not os.path.exists(os.path.join(eval_dir, 'images', 'interpolated_images',str(i))):
        os.makedirs(os.path.join(eval_dir, 'images', 'interpolated_images',str(i)))

    path = os.path.join(eval_dir, 'images', 'interpolated_images',str(i))
    # Interpolate between twe noise(z1, z2) with number_int steps between
    while number > 0:    
        number_int = 10
        z_intp = torch.FloatTensor(1, 100,1,1).to(device)
        z1 = truncated_z_sample('dcgan',1,100,0.4).to(device)
        z2 = truncated_z_sample('dcgan',1,100,0.4).to(device)
        z_intp = Variable(z_intp).to(device)
        images = []
        alpha = np.linspace(0,1,number_int)
        for i in range(1, number_int+1):
            z_intp.data = z1*alpha[i-1] + z2*(1.0 - alpha[i-1])
            fake_im = G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(3,img_size,img_size).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, path +'/interpolated_{}.png'.format(str(number).zfill(3)))
        number -=1

def get_noise(model,batch_size, noise_size, trunc, trunc_threshold ,device):
    if trunc:
        return truncated_z_sample(model, batch_size, noise_size, trunc_threshold).to(device) 
    elif model == 'dcgan':
        return torch.randn(batch_size, noise_size, 1, 1).to(device)
    else:
        return torch.randn(batch_size, noise_size).to(device)

def load_weights(model,weights):
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['g_state'])
    return model

def save_images(eval_dir, images, model_idx):
    if not os.path.exists(os.path.join(eval_dir, 'images', 'samples', str(model_idx))):
        os.makedirs(os.path.join(eval_dir, 'images', 'samples',str(model_idx)))
    image_name = os.path.join(eval_dir, 'images', 'samples',str(model_idx), 'batch.jpeg')
    vutils.save_image(images,image_name, normalize = True)
    for i in range(images.shape[0]):
            utils.save_image(images[i], os.path.join(eval_dir, 'images', 'samples',str(model_idx), str(i)+'.jpeg'),normalize= True)

parser = argparse.ArgumentParser()
# Model hyper-parameters
parser.add_argument('--no_model', type = int , default =100)
parser.add_argument('--img_size', type=int, default= 64)
parser.add_argument('--batch_size', type=int, default = 64)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--interps', type=int, default=5)
parser.add_argument('--gpus', type=str, default='0', help='gpuids eg: 0,1,2,3')
parser.add_argument('--trunc', type=bool, default = 'False')
parser.add_argument('--trunc_threshold', type=float, default = 0.02)
# Path
parser.add_argument('--eval_dir', type=str)

config = parser.parse_args()
interps = config.interps
img_size = config.img_size
batch_size = config.batch_size
eval_dir = config.eval_dir
device = config.device
trunc = config.trunc
trunc_threshold = config.trunc_threshold
model_type = eval_dir.split('_')[1]
gpus = config.gpus

if model_type == 'dcgan':
    device = torch.device(device)
    sn = 'sn' in eval_dir
    _, generator= get_dcgan(img_size, 3, sn=sn,device=device)
    noise_size = 100
else:
    gpus = [0,1,2,3]
    device = torch.device(device)
    _, generator = get_biggan(2, gpus=gpus)
    noise_size = 120

weights_dict = []
root_path = '..'
eval_dir = os.path.join(root_path, 'logs', eval_dir)
all_model_files = sorted(glob.glob(os.path.join(eval_dir, '*.pth')))
weights = []
for _w in all_model_files:
    itr = int(_w.split('/')[-1].split('_')[-1].split('.')[0])
    weights.append((itr, _w))
weights = sorted(weights, key=lambda tup: tup[0])
model_files = weights[-min(len(all_model_files) ,config.no_model):]

for i, model_fl in model_files:
    print(model_fl)
    noise  = get_noise(model_type, batch_size, noise_size, trunc, trunc_threshold, device)
    generator = load_weights(generator, model_fl)
    images = generator(noise)
    save_images(eval_dir, images, i)
    generate_latent_walk(generator, i, img_size, interps, device)