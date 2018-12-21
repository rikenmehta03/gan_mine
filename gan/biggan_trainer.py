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

from model import get_biggan


class BigGanTrainer():
    def __init__(self, num_classes, logger, log_iter=1000, d_step = 1, test_size = None, resume = None, gpus = []):
        if len(gpus) > 0:
            self.device = torch.device('cuda:' + str(gpus[0]))
        else:
            self.device = torch.device('cpu')
        self.gpus = gpus
        self.logger = logger
        self.log_iter = log_iter
        self.iter = 1
        self.num_classes = num_classes
        self.d_step = d_step
        self._build_model()
        if test_size:
            self.test_noise = self._tensor2var(torch.randn(test_size, 120)) # Input: No. of noise samples
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
        print("loaded checkpoint {} (Epoch {})".format(resume, checkpoint['iter']))
    
    def _build_model(self):
        self.discriminator, self.generator = get_biggan(self.num_classes, gpus=self.gpus)
        self.d_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=0.0002, betas = (0.0, 0.9)) #0.0004
        self.g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=0.00005, betas = (0.0, 0.9))    #0.0001

    def _denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    def _tensor2var(self, x, grad=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=grad)
    
    def _label_sample(self):
        label = torch.LongTensor(self.batch_size, 1).random_()%self.num_classes
        one_hot= torch.zeros(self.batch_size, self.num_classes).scatter_(1, label, 1)
        return label.squeeze(1).to(self.device), one_hot.to(self.device) 
    
    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        
    def train(self, data_loader, num_iter, verbose = 1):
        self.batch_size = data_loader.batch_size
        if self.test_noise is None:
            self.test_noise = self._tensor2var(torch.randn(self.batch_size, 120))
        self.data_loader = data_loader

        data_iter = iter(self.data_loader)
        start_time = time.time()
        start_iter = self.iter
        for batch_idx in range(start_iter, num_iter):
            self.generator.train()
            self.discriminator.train()
            # Train D
                
            try:
                real_batch, real_labels = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_batch, real_labels = next(data_iter)

            real_data = real_batch.to(self.device)
            real_labels = real_labels.to(self.device)
            d_out_real = self.discriminator(real_data, real_labels)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            
            # apply Gumbel Softmax
            z = torch.randn(self.batch_size, 120).to(self.device)

            z_class, z_class_one_hot = self._label_sample()
 
            fake_images = self.generator(z, z_class_one_hot)
            d_out_fake = self.discriminator(fake_images, z_class)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            z = torch.randn(self.batch_size, 120).to(self.device)
            z_class, z_class_one_hot = self._label_sample()
            
            fake_images = self.generator(z, z_class_one_hot)

            # Compute loss with fake images
            g_out_fake = self.discriminator(fake_images, z_class)  # batch x n
            g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            if verbose > 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                self.logger.print_progress((batch_idx*self.d_step) % len(self.data_loader)+1,
                    len(self.data_loader),
                    prefix = 'Train Iter: {}/{}'.format(self.iter, num_iter),
                    suffix = 'DLoss: {:.6f} GLoss: {:.6f} Elapsed: {}'.format(d_loss.item(),g_loss_fake.item(),elapsed),
                    bar_length = 50)
            
            if verbose > 0 and self.iter % 100 == 0:
                # Generate test images after training for log_iter
                test_images = self.generator(self.test_noise, z_class_one_hot)

                # Logging details.
                t_del = time.time() - start_time
                # line = '------------------ Iter: {} ------------------\n'.format(self.iter)
                # line += "Discriminator Average Error: {:.6f} , Generator Average Error: {:.6f}\n".format(d_total_error/(self.d_step*200.0),g_total_error/100.0)
                # line += 'D(x): {:.4f}, D(G(z)): {:.4f}, Time: {:.8f}\n'.format(total_pred_real/(self.d_step * 100.0 * self.batch_size),total_pred_fake/(self.d_step * 100.0 * self.batch_size), t_del)
                line = 'Item: {}, Time: {:.8f}\n'.format(self.iter, t_del)
                self.logger.log_iter(self, self.iter, line, self._denorm(test_images.data))
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