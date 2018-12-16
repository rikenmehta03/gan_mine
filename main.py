import os
import glob

from torch.backends import cudnn

from parameters import get_parameters
from data_loader import Data_Loader
from model import get_dcgan
from utils import Logger

from gan import GanTrainer, BigGanTrainer

def main(config):
    cudnn.benchmark = True

    img_size = config.img_size
    batch_size = config.batch_size
    num_iters = config.iters

    classes = None
    if config.classes is not None:
        classes = config.classes.split(',')
    elif config.dataset == 'lsun':
        raise Exception('Provide class list. Available options: bedroom_train,bridge_train,church_outdoor_train')

    data_loader = Data_Loader(config.dataset, config.data_path, img_size, batch_size, classes=classes, shuffle=True)
    logger_name = '{}_{}'.format(config.dataset, config.model)

    if config.model=='dcgan':
        from torch import nn, optim
        import torch

        sn = config.sn
        sa = config.sa
        device = torch.device(config.device)
        
        discriminator, generator = get_dcgan(img_size, 3, sn=sn, device = device)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas = (0.5, 0.999))
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas = (0.5, 0.999))
        
        if sn:
            logger_name += '_sn'
        
        if sa:
            logger_name += '_sa'

        logger = Logger(logger_name + '_{}_bs_{}'.format(img_size, batch_size))
        loss = nn.BCELoss()
        
        trainer = GanTrainer(discriminator, generator, d_optimizer, g_optimizer, loss, loss, logger, log_iter=config.log_iter, device=device)
    elif config.model == 'biggan':
        gpus = [int(x) for x in config.gpus.split(',')]
        num_classes = data_loader.num_classes
        logger = Logger(logger_name + '_{}_bs_{}'.format(img_size, batch_size))
        
        trainer = BigGanTrainer(num_classes, logger, log_iter=config.log_iter, gpus=gpus)

    trainer.train(data_loader.loader(), num_iters)
    

if __name__ == '__main__':
    config = get_parameters()
    main(config)