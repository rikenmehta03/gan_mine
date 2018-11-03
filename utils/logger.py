import os
import time 
import torchvision.utils as vutils

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class Logger():
    def __init__(self, file_name_prefix = ''):
        time_stamp = time.strftime("%d_%m_%Y_%H:%M:%S", time.localtime())
        self.dir_name = os.path.join(os.path.dirname(DIR_PATH), 'logs', time_stamp)
        
        if os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        
        self.log_file = open(os.path.join(self.dir_name, 'logfile.log'), 'w+')
        
        self.g_loss_array = []
        self.d_loss_array = []

    def _log_text(self, state):
        d_g_loss = 'Discriminator Loss: {:.4f}, Generator Loss: {:.4f}\n'.format(state['d_error'], state['g_error'])
        d_g_acc = 'D(x): {:.4f}, D(G(z)): {:.4f}\n'.format(state['d_pred_real'], state['d_pred_fake'])
        self.log_file.write('------------------ Epoch: {} ------------------\n'.format(state['epoch']))
        self.log_file.write(d_g_loss)
        self.log_file.write(d_g_acc)

    def _log_output_images(self, state, normalize=True):
        images = state['test_images']
        _grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
        
    def log(self, gan_object, state):
        self.g_loss_array.append(state['d_error'])
        self.d_loss_array.append(state['g_error'])
        
        self._log_text(state)
        self._log_output_images(state)
        