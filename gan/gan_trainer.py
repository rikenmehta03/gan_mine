import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GanTrainer():
    def __init__(self, discriminator, generator, d_optimizer, g_optimizer, d_loss, g_loss, logger):
        self.generator = generator
        self.discriminator = discriminator
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.logger = logger
        self.test_noise = self._noise(32)
    
    def _noise(self, num_samples):
        '''
        Generates a 1-d vector of gaussian sampled random values
        '''
        n = Variable(torch.randn(num_samples, self.generator.noise_size, device=device)) # add noise_size as class variable of generator
        return n
    
    def _ones_target(self, size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = Variable(torch.ones(size, 1, device=device))
        return data

    def _zeros_target(self, size):
        '''
        Tensor containing zeros, with shape = size
        '''
        data = Variable(torch.zeros(size, 1, device=device))
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
        
    def train(self, data_loader, num_epochs, batch_size, verbose = 1, checkpoint=False):
        for epoch in range(num_epochs):
            d_total_error, g_total_error = 0.0, 0.0
            total_pred_real, total_pred_fake = 0, 0
            for n_batch, (real_batch,_) in enumerate(data_loader):
                N = real_batch.size(0)

                # 1. Train Discriminator
                real_data = Variable(real_batch).to(device)

                # Generate fake data and detach 
                # (so gradients are not calculated for generator)
                fake_data = self.generator(self._noise(N)).detach().to(device)

                # Train D
                d_error, d_pred_real, d_pred_fake =  self._train_discriminator(real_data, fake_data)

                # 2. Train Generator

                # Generate fake data
                fake_data = self.generator(self._noise(N)).to(device)

                # Train G
                g_error = self._train_generator(fake_data)

                d_total_error += d_error
                g_total_error += g_error
                total_pred_real += d_pred_real.sum()
                total_pred_fake += d_pred_fake.sum()

            test_images = self.generator(self.test_noise)

            d_total_error = (d_total_error * batch_size)/(2 * len(data_loader.dataset))
            g_total_error = (g_total_error * batch_size)/(len(data_loader.dataset))
            total_pred_fake /= (1.0 * len(data_loader.dataset))
            total_pred_real /= (1.0 * len(data_loader.dataset)) 
            
            state = {
                'epoch': epoch,
                'd_error': d_total_error,
                'g_error': g_total_error,
                'd_pred_real': total_pred_real,
                'd_pred_fake': total_pred_fake,
                'test_images': test_images
            }
            self.logger.log(self, state)
