import os, sys

import torch
from data_loader import Data_Loader
from evaluation_metric import Evaluator
from model import get_biggan, get_dcgan
from parameters import get_parameters

def main(config):
    img_size = config.img_size
    batch_size = config.batch_size
    eval_folder = config.eval_folder
    device = config.device
    classes = None
    if config.classes is not None:
        classes = config.classes.split(',')
    elif config.dataset == 'lsun':
        raise Exception('Provide class list. Available options: bedroom_train,bridge_train,church_outdoor_train')
    
    model_type = eval_folder.split('_')[1]
    data_loader = Data_Loader(config.dataset, config.data_path, img_size, batch_size, classes=classes, shuffle=False)
    if model_type == 'dcgan':
        device = torch.device(device)
        sn = 'sn' in eval_folder
        _, generator= get_dcgan(img_size, 3, sn=sn,device=device)
    else:
        gpus = [int(device.split(':')[-1])]
        device = torch.device(device)
        _, generator = get_biggan(data_loader.num_classes, gpus=gpus)
    
    evaluator = Evaluator(generator, data_loader, eval_folder, device, batch_size=batch_size)
    evaluator.run()

if __name__ == "__main__":
    config = get_parameters()
    main(config)

