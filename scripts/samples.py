import argparse
from model import get_biggan, get_dcgan
import os
import glob
import torch
import numpy as np

def truncated_z_sample(batch_size, truncation=1., seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
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

def _save_images(eval_dir, images, batch_idx):
    if not os.path.exists(os.path.join(eval_dir, 'images', 'samples')):
        os.makedirs(os.path.join(eval_dir, 'images', dataset))
    
    for i in range(images.shape[0]):
            idx = i + batch_idx*self.batch_size
            utils.save_image(images[i], os.path.join(self.eval_dir, 'images', self.dataset, str(idx)+'.jpeg'))

parser = argparse.ArgumentParser()
# Model hyper-parameters
parser.add_argument('--no_model', type = int , default =100)
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default = 32)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gpus', type=str, default='0', help='gpuids eg: 0,1,2,3')

# Path
parser.add_argument('--eval_folder', type=str)

config = parser.parse_args()

img_size = config.img_size
batch_size = config.batch_size
eval_folder = config.eval_folder
device = config.device
model_type = eval_folder.split('_')[1]
if model_type == 'dcgan':
    device = torch.device(device)
    sn = 'sn' in eval_folder
    _, generator= get_dcgan(img_size, 3, sn=sn,device=device)
else:
    gpus = [int(device.split(':')[-1])]
    device = torch.device(device)
    _, generator = get_biggan(2, gpus=gpus)

weights_dict = []
root_path = os.path.dirname(os.path.dirname(__file__))
eval_dir = os.path.join(root_path, 'logs', eval_folder)
model_files = sorted(glob.glob(os.path.join(eval_dir, '*.pth')))[-config.no_model:]

for i, model_fl in enumerate(model_files):
    noise  = get_noise(batch_size,device)
    generator = load_weights(generator, model_fl)
    images = generator(noise)
