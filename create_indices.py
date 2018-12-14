import os, sys
import json
import argparse

import nmslib
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import utils
from data_loader import Data_Loader
from evaluation_metric.inception import InceptionV3

# Global: nmslib index parameters
M = 100
efC = 2000
num_threads = 4
space_name='l2'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'celebhq', 'celeb', 'imagenet', 'lsun'])
parser.add_argument('--path', type=str, default='dataset/')
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--out', type=str, default='indices/')

def get_index():
    index = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
    return index

def create_index(index, path):
    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC}
    index.createIndex(index_time_params) 
    index.saveIndex(path)

def save_images(images, labels, batch_idx, batch_size, class_dict, path):
    if not os.path.exists(os.path.join(path, 'images')):
        os.makedirs(os.path.join(path, 'images'))

    for i in range(images.shape[0]):
        idx = i + batch_idx*batch_size
        utils.save_image(images[i], os.path.join(path, 'images', str(idx)+'.jpeg'))
        class_dict[idx] = labels[i].item()
    
    return class_dict

def main(config):
    dataset = config.dataset
    path = config.path
    img_size = config.img_size
    out_dir = os.path.join(config.out, dataset)
    batch_size = config.batch_size
    if dataset == 'lsun':
        dataloader = Data_Loader(dataset, path, img_size, batch_size, classes=['bedroom_train', 'bridge_train', 'church_outdoor_train'], shuffle=False)
    else:
        dataloader = Data_Loader(dataset, path, img_size, batch_size, shuffle=False)
    
    class_dict = {}

    device = torch.device(config.device)
    model = InceptionV3().to(device)

    data_iter = dataloader.loader()

    index = get_index()    
    
    for batch_idx, (images, labels) in enumerate(data_iter):
        batch = images.to(device)
        preds = model(batch)[0]
        preds = preds.numpy().reshape((preds.shape[0], preds.shape[1]))
        index.addDataPointBatch(preds, range(batch_idx*batch_size, (batch_idx+1)*batch_size))
        class_dict = save_images(images, labels, batch_idx, batch_size, class_dict, out_dir)
        
    create_index(index, os.path.join(out_dir, 'index.bin'))
    with open(os.path.join(out_dir, 'class_dict.json'), 'w') as fp:
        json.dump(class_dict, fp)

if __name__ == "__main__":
    main(parser.parse_args())