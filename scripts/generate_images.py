import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gan_mine.evaluation_metric import inception_score
from gan_mine.data_loader import Data_Loader
from gan_mine.model import get_dcgan
from torchvision import utils

batch_size = 64
num_images = 50048
num_batches = int(num_images/ batch_size)

data = 'celeb'
model_path = 'gan_mine/logs/celeb_dcgan_64_bs_32_16_12_2018_22:08:01/model_40500.pth' 

import torch
device = torch.device('cuda:1')
_, generator = get_dcgan(64, 3, sn=False, device=device)
checkpoint = torch.load(model_path)
generator.load_state_dict(checkpoint['g_state'])

def save_images(images, batch_idx):
    if not os.path.exists(os.path.join(data+'_images', 'images', data)):
        os.makedirs(os.path.join(data+'_images', 'images', data))

    for i in range(images.shape[0]):
        idx = i + batch_idx*batch_size
        utils.save_image(images[i], os.path.join(data+'_images', 'images', data, str(idx)+'.jpeg'))

for batch_idx in range(num_batches):
    start = batch_idx*64
    end = (batch_idx+1)*batch_size
    noise = torch.randn(64, 100, 1, 1, device=device)
    images = generator(noise)
    save_images(images, batch_idx)