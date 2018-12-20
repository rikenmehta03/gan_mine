import os, sys
import time
import datetime
import errno
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import StatisticNetworkDCGAN

class MINE_trainer():
    
    def __init__(self, discriminator, generator,stat_net , d_optimizer, g_optimizer, s_optimizer, d_loss, g_loss, logger, log_iter=1000, test_size = None, resume = None, device = torch.device('cpu')):
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.stat_net = stat_net
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.s_optimizer = s_optimizer
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.logger = logger
        self.log_iter = log_iter
        self.iter = 1
        self.noise_size = self.generator.in_ch
        if test_size:
            self.test_noise = torch.randn(test_size, self.noise_size, 1, 1).to(self.device) # Input: No. of noise samples
        else:
            self.test_noise = None

        if resume is not None:
            if os.path.exists(resume):
                self._load_checkpoint(resume)
                os.rmdir(self.logger.dir_name)
                dirs = self.logger.dir_name.split('/')
                dirs[-1] = resume.split('/')[-2]
                self.logger.dir_name = '/' + os.path.join(*dirs)
                self.logger.log_file_name = os.path.join(self.logger.dir_name, 'logfile.log')
            else:
                raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), resume)

    def _denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    
    def _load_checkpoint(self, resume):
        '''
        Loads a given checkpoint. Throws an error if the path doesn't exist.

        Input: File path
        '''
        checkpoint = torch.load(resume)
        self.iter = checkpoint['iter']
        self.generator.load_state_dict(checkpoint['g_state'])
        self.discriminator.load_state_dict(checkpoint['d_state'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        print("loaded checkpoint {} (Iter {})".format(resume, checkpoint['iter']))

        
    def train(self, data_loader, num_iter, verbose = 1):
        
        #b minibatches
            # T(theta) is output of statistics network
            # lower bound = E[T(theta)] - log(E[e^T(theta)])
            # gradient_lb  = E[grad_theta*T(theta)]{x,y} - E[grad_theta * T(theta)/E[e^T(theta)]]{x,y~}
            # fix biased gradients.
            # New_E[e^T(theta)]_i = alpha * E[e^T(theta)] + (1 - alpha) * New_E[e^T(theta)]_(i-1) { when i = 1, then New_E[e^T(theta)]_1 = E[e^T(theta)]_(batch 1) }
            # Then,
            # Fixed_gradient_lb = E[grad * T(theta)] - E[grad_theta * T(theta)/New_E[e^T(theta)]
            # theta = theta + Fixed_gradient_lb * gradient_lb  
        
        self.batch_size = data_loader.batch_size
        if self.test_noise is None:
            self.test_noise = torch.randn(self.batch_size, self.generator.in_ch, 1, 1).to(self.device)
        
        self.data_loader = data_loader
        
        data_iter = iter(self.data_loader)
        start_time = time.time()
        start_iter = self.iter
        ones = torch.ones(self.batch_size, 1).to(self.device)
        zeros = torch.zeros(self.batch_size, 1).to(self.device)

        for batch_idx in range(start_iter, num_iter):
            
            self.generator.train()
            self.discriminator.train()
            self.stat_net.train()
            
            try:
                real_batch, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_batch, _ = next(data_iter)

            real_data = real_batch.to(self.device)

            # Train D    
            z = torch.randn(self.batch_size, self.noise_size, 1, 1).to(self.device)        
            fake_data = self.generator(z).detach().to(self.device)

            self.d_optimizer.zero_grad()

            d_out_real = self.discriminator(real_data)
            d_loss_real = self.d_loss(d_out_real, ones)
            d_loss_real.backward()

            d_out_fake = self.discriminator(fake_data)
            d_loss_fake = self.d_loss(d_out_fake, zeros)
            d_loss_fake.backward()

            self.d_optimizer.step()
            d_loss = d_loss_real + d_loss_real

            # ================== Train G ================== #
            self.g_optimizer.zero_grad()

            fake_data = self.generator(z).to(self.device)
            g_out_fake = self.discriminator(fake_data)  # batch x n
            g_loss_fake = self.g_loss(g_out_fake, ones)

            g_loss_fake.backward()
            self.g_optimizer.step()

            if verbose > 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                self.logger.print_progress((batch_idx) % len(self.data_loader)+1,
                    len(self.data_loader),
                    prefix = 'Train Iter: {}/{}'.format(self.iter, num_iter),
                    suffix = 'DLoss: {:.6f} GLoss: {:.6f} Elapsed: {}'.format(d_loss.item(),g_loss_fake.item(),elapsed),
                    bar_length = 50)
            
            if verbose > 0 and self.iter % 100 == 0:
                test_images = self.generator(self.test_noise).detach()
                t_del = time.time() - start_time
                line = 'Item: {}, Time: {:.8f}\n'.format(self.iter, t_del)
                self.logger.log_iter(self, self.iter, line, self._denorm(test_images.data), normalize=True)
                del test_images

            if self.iter % self.log_iter == 0:
                state = {
                    'iter': self.iter,
                    'd_state': self.discriminator.state_dict(),
                    'g_state': self.generator.state_dict(),
                    'd_optimizer': self.d_optimizer.state_dict(),
                    'g_optimizer': self.g_optimizer.state_dict()
                }
                self.logger.log_epoch(self, state)
            
            self.iter += 1