import os, sys
import shutil
import glob
import json

import torch
from torchvision import utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader import Data_Loader
from .inception_score import inception_score
from .fid_score import fid_score

root_path = os.path.dirname(os.path.dirname(__file__))

class Evaluator():
    def __init__(self, model, dataloader, path, device, image_count=10240, batch_size=64):
        self.eval_dir = os.path.join(root_path, 'logs', path)
        self.model_type = path.split('_')[1]
        self.model = model
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        self.weights_dict = {}
        self.image_count = image_count
        self.device = device
        self.batch_size = batch_size
        
        for _w in glob.glob(os.path.join(self.eval_dir, '*.pth')):
            itr = int(_w.split('/')[-1].split('_')[-1].split('.')[0])
            self.weights_dict[itr] = _w
        
        self.summary = {}

        if os.path.exists(os.path.join(self.eval_dir, 'summary.json')):
            with open(os.path.join(self.eval_dir, 'summary.json'), 'r') as fp:
                self.summary = json.load(fp)
        
        if self.summary.get('image_count', 0) != image_count:
            self.summary = {
                'inception_done' : [],
                'fid_done': [],
                'inception_score': [],
                'fid': [],
                'image_count': image_count
            }
        
        self._init_noise()

    def _load_weights(self, weights):
        checkpoint = torch.load(weights)
        self.iter = checkpoint['iter']
        self.model.load_state_dict(checkpoint['g_state'])

    def _init_noise(self):
        if self.model_type == 'dcgan':
            self.noise = torch.randn(self.image_count, 100, 1, 1, device=self.device)
        else:
            self.noise = torch.randn(self.image_count, 120, device=self.device)
    
    def _save_images(self, images, batch_idx):
        if not os.path.exists(os.path.join(self.eval_dir, 'images', self.dataset)):
            os.makedirs(os.path.join(self.eval_dir, 'images', self.dataset))

        for i in range(images.shape[0]):
            idx = i + batch_idx*self.batch_size
            utils.save_image(images[i], os.path.join(self.eval_dir, 'images', self.dataset, str(idx)+'.jpeg'))
    
    def _label_sampel(self, num_classes):
        label = torch.LongTensor(self.batch_size, 1).random_()%num_classes
        one_hot= torch.zeros(self.batch_size, num_classes).scatter_(1, label, 1)
        return label.squeeze(1).to(self.device), one_hot.to(self.device) 

    def _generate(self, weights):
        self._load_weights(weights)
        num_batches = int(self.image_count / self.batch_size)
        
        for batch_idx in range(num_batches):
            start = batch_idx*self.batch_size
            end = (batch_idx+1)*self.batch_size
            noise = self.noise[start:end]
            if self.model_type == 'dcgan':
                images = self.model(noise).detach()
            else:
                num_classes = self.dataloader.num_classes
                _, z_class_one_hot = self._label_sampel(num_classes)
                images = self.model(noise, z_class_one_hot).detach()
            self._save_images(images, batch_idx)

    def _find_is(self):
        data_loader = Data_Loader('images', self.eval_dir, self.dataloader.imsize, self.batch_size, shuffle=False)
        _is = inception_score(data_loader.loader(), self.device, True, 10)
        self.summary['inception_score'].append((self.iter, _is))
        self.summary['inception_done'].append(self.iter)
    
    def _find_fid(self):
        data_loader = Data_Loader('images', self.eval_dir, self.dataloader.imsize, self.batch_size, shuffle=False)
        _fid = fid_score(self.dataloader.loader(), self.dataset, data_loader.loader(), device=self.device)
        self.summary['fid'].append((self.iter, _fid))
        self.summary['fid_done'].append(self.iter)
    
    def run(self):
        weights_iter = sorted(self.weights_dict.items(), key=lambda tup: tup[0])
        for itr, weights in weights_iter:
            try:
                shutil.rmtree(os.path.join(self.eval_dir, 'images'))
            except:
                pass

            find_is = itr not in self.summary['inception_done']
            find_fid = itr not in self.summary['fid_done']

            if find_is or find_fid:
                self._generate(weights)

            if find_is:
                self._find_is()

            if find_fid:
                self._find_fid()

            with open(os.path.join(self.eval_dir, 'summary.json'), 'w') as fp:
                json.dump(self.summary, fp)

        best_inception_score = sorted(self.summary['inception_score'], key=lambda k: k[1][0])[-1]
        best_fid_score = sorted(self.summary['fid'], key=lambda k: k[1])[0]
        with open(os.path.join(self.eval_dir, 'summary.txt'), 'w') as fp:
            s = 'Inception score: {}\n'.format(best_inception_score[1][0])
            s += 'at Iteration: {}\n\n'.format(best_inception_score[0])
            s += 'Fid: {}\n'.format(best_fid_score[1])
            s += 'at Iteration: {}'.format(best_fid_score[0])
            fp.write(s)
        
        self.summary['best_fid'] = best_fid_score
        self.summary['best_inception'] = best_inception_score
        with open(os.path.join(self.eval_dir, 'summary.json'), 'w') as fp:
            json.dump(self.summary, fp)



        