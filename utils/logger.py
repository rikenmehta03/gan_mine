import os
import sys
import time 
import copy
import torch
import torchvision.utils as vutils

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class Logger():
    def __init__(self, file_name_prefix = None):
        time_stamp = time.strftime("%d_%m_%Y_%H:%M:%S", time.localtime())
        if file_name_prefix is not None:
            time_stamp = file_name_prefix + '_' + time_stamp
        self.dir_name = os.path.join(os.path.dirname(DIR_PATH), 'logs', time_stamp)
        
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        
        self.log_file_name = os.path.join(self.dir_name, 'logfile.log')

        self.g_loss_array = []
        self.d_loss_array = []

    def _log_text(self, state):
        d_g_loss = 'Discriminator Loss: {:.4f}, Generator Loss: {:.4f}\n'.format(state['d_error'], state['g_error'])
        d_g_acc = 'D(x): {:.4f}, D(G(z)): {:.4f}\n'.format(state['d_pred_real'], state['d_pred_fake'])
        with open(self.log_file_name, 'a') as fp:
            fp.write('------------------ Epoch: {} ------------------\n'.format(state['epoch']))
            fp.write(d_g_loss)
            fp.write(d_g_acc)
    
    def _model_checkpoint(self, state):
        filename = os.path.join(self.dir_name, 'model_{}.pth'.format(state['epoch']))
        torch.save(state, filename)

    def _log_output_images(self, state, images, normalize=True):
        _grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
        image_name = os.path.join(self.dir_name, str(state['epoch']) +'_GIMG.png')
        vutils.save_image(_grid,image_name) 

    def log(self, gan_object, state, test_images):
        self.g_loss_array.append(state['d_error'])
        self.d_loss_array.append(state['g_error'])
        
        self._log_text(state)
        self._log_output_images(state, test_images)
        self._model_checkpoint(state)
    
    def print_progress(self, iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(bar_length * iteration / float(total)))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

