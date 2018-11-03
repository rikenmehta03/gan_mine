import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class GanTrainer():
    def __init__(self, discriminator, generator, d_optimizer, g_optimizer, d_loss, g_loss, logger, device = torch.device('cpu')):
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.logger = logger
        self.test_noise = self._noise(32) # Input: No. of noise samples
    
    def _noise(self, num_samples):
        '''
        Generates a 1-d vector of gaussian sampled random values
        '''
        n = torch.randn(num_samples, self.generator.noise_size, 1, 1, device = self.device) # add noise_size as class variable of generator
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

        
        #--------------------------------------------------------------------------------------------
        # Training Ends.
        # -------------------------------------------------------------------------------------------
        
        # Generate test images after training for each epoch

        test_images = self.generator(self.test_noise)

        # Logging details.

        d_total_error = (d_total_error * data_loader.batch_size)/(2 * len(data_loader.dataset))
        g_total_error = (g_total_error * data_loader.batch_size)/(len(data_loader.dataset))
        total_pred_fake /= (1.0 * len(data_loader.dataset))
        total_pred_real /= (1.0 * len(data_loader.dataset))

        if verbose > 0:
            line = '------------------ Epoch: {} ------------------\n'.format(epoch)
            line += "Discriminator Average Error: {:.6f} , Generator Average Error: {:.6f}\n".format(d_total_error,g_total_error)
            line += 'D(x): {:.4f}, D(G(z)): {:.4f}\n'.format(total_pred_real,total_pred_fake)
            print(line)
        
        state = {
            'epoch': epoch,
            'd_error': d_total_error,
            'g_error': g_total_error,
            'd_pred_real': total_pred_real,
            'd_pred_fake': total_pred_fake,
            'test_images': test_images
        }
        return state
        #self.logger.log(self, state)

    def trainer(self, data_loader, num_epochs, verbose = 1, checkpoint=False):
        for epoch in range(1, num_epochs+1):
            state = self._train(epoch, data_loader, verbose)
            self.logger.log(self, state)


