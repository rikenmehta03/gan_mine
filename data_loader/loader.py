import os
import torch
from torchvision import transforms, datasets

class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

class Data_Loader():
    def __init__(self, dataset, image_path, image_size, batch_size, shuffle=True):
        self.dataset = dataset

        if 'load_' + self.dataset not in dir(self):
            raise InputError(self.dataset, 'Select valid dataset. Available options: [lsun, celeb, celebhq, cifar10, cifar100, imagenet]')

        self.path = image_path

        if self.dataset == 'lsun':
            assert os.path.exists(os.path.join(self.path, 'bedroom_train_lmdb'))
        elif 'cifar' not in self.dataset:
            assert os.path.exists(os.path.join(self.path, self.dataset))

        self.imsize = image_size
        self.batch = batch_size
        self.shuffle = shuffle

    def transform(self, resize, totensor, normalize):
        options = []
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes=['bedroom_train']):
        transforms = self.transform(True, True, True)
        dataset = datasets.LSUN(self.path, classes=classes, transform=transforms)
        return dataset
    
    def load_imagenet(self):
        transforms = self.transform(True, True, True)
        dataset = datasets.ImageFolder(self.path+'/imagenet', transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True)
        dataset = datasets.ImageFolder(self.path+'/celeb', transform=transforms)
        return dataset
    
    def load_celebhq(self):
        transforms = self.transform(True, True, True)
        dataset = datasets.ImageFolder(self.path+'/celebhq', transform=transforms)
        return dataset

    def load_cifar10(self):
        transforms = self.transform(True, True, True)
        dataset = datasets.CIFAR10(self.path, transform=transforms)
        return dataset
    
    def load_cifar100(self):
        transforms = self.transform(True, True, True)
        dataset = datasets.CIFAR100(self.path, transform=transforms)
        return dataset

    def loader(self):

        dataset = getattr(self, 'load_' + self.dataset)()
        print('dataset',len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuffle,
                                              num_workers=4)
        return dataloader
    
    @classmethod
    def from_dict(cls, params):
        return cls(params['dataset'], params['data_dir'], params['img_size'], params['batch_size'])