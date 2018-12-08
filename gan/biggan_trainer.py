import os
import time
import datetime
import errno
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class BigGanTrainer():
    def __init__(self, discriminator, generator, d_optimizer, g_optimizer, num_classes, logger, log_iter=1000, d_step = 1, test_size = None, resume = None, device = torch.device('cpu')):
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.logger = logger
        self.log_iter = log_iter
        self.iter = 1
        self.num_classes = num_classes
        self.d_step = d_step
        if test_size:
            self.test_noise = self._noise(test_size) # Input: No. of noise samples
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
        print("loaded checkpoint {} (Epoch {})".format(resume, checkpoint['epoch']))

    def _noise(self, num_samples):
        '''
        Generates a 1-d vector of gaussian sampled random values
        '''
        n = torch.randn(num_samples, 120, device = self.device) 
        return n
    
    def _denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    def _label_sampel(self):
        label = torch.LongTensor(self.batch_size, 1).random_()%self.num_classes
        one_hot= torch.zeros(self.batch_size, self.num_classes).scatter_(1, label, 1)
        return label.squeeze(1).to(self.device), one_hot.to(self.device) 
    
    def _train_discriminator(self, real_data, real_labels):
        N = real_data.size(0)
        
        # 1.1 Train on Real Data
        prediction_real = self.discriminator(real_data, real_labels)
        error_real = torch.nn.ReLU()(1.0 - prediction_real).mean()

        # 1.2 Train on Fake Data
        z_class, z_class_one_hot = self._label_sampel()
        fake_data = self.generator(self._noise(N), z_class_one_hot).detach().to(self.device)
        
        prediction_fake = self.discriminator(fake_data, z_class)
        # Calculate error and backpropagate
        error_fake = torch.nn.ReLU()(1.0 + prediction_fake).mean()
        
        # 1.3 Update weights with gradients
        error = error_fake + error_real
        self.d_optimizer.zero_grad()
        error.backward()
        self.d_optimizer.step()
        
        # Return error and predictions for real and fake inputs
        return error, prediction_real, prediction_fake
    
    def _train_generator(self):
        N = self.batch_size
        z_class, z_class_one_hot = self._label_sampel()
        fake_data = self.generator(self._noise(N), z_class_one_hot).detach().to(self.device)

        # Reset gradients
        self.g_optimizer.zero_grad()

        # Sample noise and generate fake data
        prediction = self.discriminator(fake_data, z_class)

        # Calculate error and backpropagate
        error = - prediction.mean()
        error.backward()

        # Update weights with gradients
        self.g_optimizer.step()

        # Return error
        return error
        
    def _train(self, num_iter, verbose):
        data_iter = iter(self.data_loader)
        start_time = time.time()
        d_total_error, g_total_error = 0.0, 0.0
        total_pred_real, total_pred_fake = 0, 0
        start_iter = self.iter
        for batch_idx in range(start_iter, num_iter):
            self.generator.train()
            self.discriminator.train()
            # Train D
            d_error, d_pred_real, d_pred_fake  = 0, 0, 0 
            for _ in range(self.d_step):
                
                try:
                    real_batch, real_labels = next(data_iter)
                except:
                    data_iter = iter(self.data_loader)
                    real_batch, real_labels = next(data_iter)

                real_data = real_batch.to(self.device)
                real_labels = real_labels.to(self.device)

                _d_error, _d_pred_real, _d_pred_fake =  self._train_discriminator(real_data, real_labels)
                d_error += _d_error.item()
                d_pred_real += _d_pred_real.sum()
                d_pred_fake += _d_pred_fake.sum()
                
            # Train G
            g_error = self._train_generator().item()

            d_total_error += d_error
            total_pred_real += d_pred_real
            total_pred_fake += d_pred_fake
            g_total_error += g_error
            

            if verbose > 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                self.logger.print_progress((batch_idx*self.d_step) % len(self.data_loader),
                    len(self.data_loader),
                    prefix = 'Train Iter: {}/{}'.format(self.iter, num_iter),
                    suffix = 'DLoss: {:.6f} GLoss: {:.6f} Elapsed: {}'.format(d_error/self.d_step,g_error,elapsed),
                    bar_length = 50)
            
            if verbose > 0 and self.iter % 100 == 0:
                # Generate test images after training for log_iter
                _, z_class_one_hot = self._label_sampel()
                test_images = self.generator(self.test_noise, z_class_one_hot).detach()

                # Logging details.
                t_del = time.time() - start_time
                line = '------------------ Iter: {} ------------------\n'.format(self.iter)
                line += "Discriminator Average Error: {:.6f} , Generator Average Error: {:.6f}\n".format(d_total_error/(self.d_step*200.0),g_total_error/100.0)
                line += 'D(x): {:.4f}, D(G(z)): {:.4f}, Time: {:.8f}\n'.format(total_pred_real/(self.d_step * 100.0 * self.batch_size),total_pred_fake/(self.d_step * 100.0 * self.batch_size), t_del)
                self.logger.log_iter(self, self.iter, line, self._denorm(test_images))
                d_total_error, g_total_error = 0.0, 0.0
                total_pred_real, total_pred_fake = 0, 0
            
            if self.iter % self.log_iter == 0:
                state = {
                    'iter': self.iter,
                    'd_error': d_total_error,
                    'g_error': g_total_error,
                    'd_pred_real': total_pred_real,
                    'd_pred_fake': total_pred_fake,
                    'd_state': self.discriminator.state_dict(),
                    'g_state': self.generator.state_dict(),
                    'd_optimizer': self.d_optimizer.state_dict(),
                    'g_optimizer': self.g_optimizer.state_dict()
                }
                self.logger.log_epoch(self, state)
            
            self.iter += 1

    def trainer(self, data_loader, num_iter, verbose = 1, checkpoint=False):
        if self.test_noise is None:
            self.test_noise = self._noise(data_loader.batch_size)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self._train(num_iter, verbose)
        


