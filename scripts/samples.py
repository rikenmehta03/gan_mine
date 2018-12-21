
import argparse
import os
import glob
import torch
import numpy as np
import sys
from scipy.stats import truncnorm
from torchvision import utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import get_biggan, get_dcgan

def truncated_z_sample(batch_size, truncation=1., seed=None):
  state = None
  values = truncnorm.rvs(-2, 2, size=(batch_size, 120), random_state = state)
  return truncation * values

def interpolate(A, B, num_interps):
  alphas = np.linspace(0, 1, num_interps)
  if A.shape != B.shape:
    raise ValueError('A and B must have the same shape to interpolate.')
  return np.array([(1-a)*A + a*B for a in alphas])

def get_noise(batch_size,device):
    return torch.randn(batch_size, 120).to(device)

def load_weights(model,weights):
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['g_state'])
    return model

def save_images(eval_dir, images, model_idx):
    if not os.path.exists(os.path.join(eval_dir, 'images', 'samples', str(model_idx))):
        os.makedirs(os.path.join(eval_dir, 'images', 'samples',str(model_idx)))
    
    for i in range(images.shape[0]):
            utils.save_image(images[i], os.path.join(eval_dir, 'images', 'samples', str(i)+'.jpeg'))

parser = argparse.ArgumentParser()
# Model hyper-parameters
parser.add_argument('--no_model', type = int , default =100)
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default = 32)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gpus', type=str, default='0', help='gpuids eg: 0,1,2,3')

# Path
parser.add_argument('--eval_dir', type=str)

config = parser.parse_args()

img_size = config.img_size
batch_size = config.batch_size
eval_dir = config.eval_dir
device = config.device
model_type = eval_dir.split('_')[1]
if model_type == 'dcgan':
    device = torch.device(device)
    sn = 'sn' in eval_dir
    _, generator= get_dcgan(img_size, 3, sn=sn,device=device)
else:
    gpus = [int(device.split(':')[-1])]
    device = torch.device(device)
    _, generator = get_biggan(2, gpus=gpus)

weights_dict = []
root_path = os.path.dirname(__file__)
eval_dir = os.path.join(root_path, 'logs', eval_dir)
all_model_files = sorted(glob.glob(os.path.join(eval_dir, '*.pth')))
model_files = all_model_files[-min(len(all_model_files) ,config.no_model):]
print(root_path)
print(eval_dir)
for i, model_fl in enumerate(model_files):
    print(i)
    noise  = get_noise(batch_size,device)
    generator = load_weights(generator, model_fl)
    images = generator(noise)
    save_images(eval_dir, images, i)
