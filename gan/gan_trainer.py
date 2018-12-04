import time
import os
import errno
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class GanTrainer():
    def __init__(self, discriminator, generator, d_optimizer, g_optimizer, d_loss, g_loss, logger, test_size = None, resume = None, device = torch.device('cpu')):
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.logger = logger
        self.start_epoch = 1
        self.iter = 1
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
        self.start_epoch = checkpoint['epoch'] + 1
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
        n = torch.randn(num_samples, self.generator.in_ch, 1, 1, device = self.device) 
        return n
    
    def _ones_target(self, size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = torch.ones(size, 1, device = self.device)
        return data

    def _zeros_target(self, size):
        '''
        Tensor containing zeros, with shape = size
        '''
        data = torch.zeros(size, 1, device = self.device)
        return data
    
    def _train_discriminator(self, real_data, fake_data):
        N = real_data.size(0)
        # Reset gradients
        self.d_optimizer.zero_grad()
        
        # 1.1 Train on Real Data
        prediction_real = self.discriminator(real_data)
        # Calculate error and backpropagate
        error_real = self.d_loss(prediction_real, self._ones_target(N))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = self.discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = self.d_loss(prediction_fake, self._zeros_target(N))
        error_fake.backward()
        
        # 1.3 Update weights with gradients
        self.d_optimizer.step()
        
        # Return error and predictions for real and fake inputs
        return error_real + error_fake, prediction_real, prediction_fake
    
    def _train_generator(self, fake_data):
        N = fake_data.size(0)

        # Reset gradients
        self.g_optimizer.zero_grad()

        # Sample noise and generate fake data
        prediction = self.discriminator(fake_data)

        # Calculate error and backpropagate
        error = self.g_loss(prediction, self._ones_target(N))
        error.backward()

        # Update weights with gradients
        self.g_optimizer.step()

        # Return error
        return error
        
    def _train(self, epoch, data_loader, verbose):
        start_time = time.time()
        d_total_error, g_total_error = 0.0, 0.0
        total_pred_real, total_pred_fake = 0, 0
        for batch_idx, (real_batch,_) in enumerate(data_loader):
            N = real_batch.size(0)

            # 1. Train Discriminator
            real_data = real_batch.to(self.device)

            # Generate fake data and detach 
            # (so gradients are not calculated for generator)
            fake_data = self.generator(self._noise(N)).detach().to(self.device)

            # Train D
            d_error, d_pred_real, d_pred_fake =  self._train_discriminator(real_data, fake_data)

            # 2. Train Generator

            # Generate fake data
            fake_data = self.generator(self._noise(N)).to(self.device)

            # Train G
            g_error = self._train_generator(fake_data)

            d_total_error += d_error
            g_total_error += g_error
            total_pred_real += d_pred_real.sum()
            total_pred_fake += d_pred_fake.sum()

            if verbose > 0:
                self.logger.print_progress(batch_idx + 1,
                    len(data_loader),
                    prefix = 'Train Epoch: {}'.format(epoch),
                    suffix = 'DLoss: {:.6f} GLoss: {:.6f}'.format(d_error.item(),g_error.item()),
                    bar_length = 50)
            
            if verbose > 0 and self.iter % 100 == 0:
                # Generate test images after training for each epoch
                self.generator.eval()
                test_images = self.generator(self.test_noise)
                self.generator.train()
                # Logging details.
                t_del = time.time() - start_time
                line = '------------------ Iter: {} ------------------\n'.format(self.iter)
                line += "Discriminator Average Error: {:.6f} , Generator Average Error: {:.6f}\n".format(d_total_error/200.0,g_total_error/100.0)
                line += 'D(x): {:.4f}, D(G(z)): {:.4f}, Time: {:.8f}\n'.format(total_pred_real/(100.0 * data_loader.batch_size),total_pred_fake/(100.0 * data_loader.batch_size), t_del)
                self.logger.log_iter(self, self.iter, line, test_images)
                d_total_error, g_total_error = 0.0, 0.0
                total_pred_real, total_pred_fake = 0, 0
            
            self.iter += 1

        
        #--------------------------------------------------------------------------------------------
        # Training Ends.
        # -------------------------------------------------------------------------------------------
        
        # # Generate test images after training for each epoch
        # self.generator.eval()
        # test_images = self.generator(self.test_noise)
        # self.generator.train()
        # # Logging details.

        # d_total_error = (d_total_error * data_loader.batch_size)/(2 * len(data_loader.dataset))
        # g_total_error = (g_total_error * data_loader.batch_size)/(len(data_loader.dataset))
        # total_pred_fake /= (1.0 * len(data_loader.dataset))
        # total_pred_real /= (1.0 * len(data_loader.dataset))
        # t_del = time.time() - start_time
        # if verbose > 0:
        #     line = '------------------ Epoch: {} ------------------\n'.format(epoch)
        #     line += "Discriminator Average Error: {:.6f} , Generator Average Error: {:.6f}\n".format(d_total_error,g_total_error)
        #     line += 'D(x): {:.4f}, D(G(z)): {:.4f}, Time: {:.8f}\n'.format(total_pred_real,total_pred_fake, t_del)
        #     print(line)
        
        state = {
            'epoch': epoch,
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
        return state

    def trainer(self, data_loader, num_epochs, verbose = 1, checkpoint=False):
        if self.test_noise is None:
            self.test_noise = self._noise(data_loader.batch_size)
        for epoch in range(self.start_epoch, num_epochs+1):
            state = self._train(epoch, data_loader, verbose)
            self.logger.log_epoch(self, state)


