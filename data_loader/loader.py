import os
import glob
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
    def __init__(self, dataset, image_path, image_size, batch_size, classes=None, shuffle=True):
        self.dataset = dataset

        # if 'load_' + self.dataset not in dir(self):
        #     raise InputError(self.dataset, 'Select valid dataset. Available options: [lsun, celeb, celebhq, cifar10, cifar100, imagenet]')

        self.path = image_path

        if self.dataset == 'lsun':
            if classes is None:
                raise InputError(classes, 'Provide class list. Available options: [bedroom_train, bridge_train, church_outdoor_train]')
            
            for _c in classes:
                assert os.path.exists(os.path.join(self.path, _c + '_lmdb'))
            self.classes = classes
            self.num_classes = len(self.classes)
        elif 'cifar' not in self.dataset:
            assert os.path.exists(os.path.join(self.path, self.dataset))
            self.num_classes = len(glob.glob(os.path.join(self.path, self.dataset, '*')))
        else:
            self.num_classes = int(self.dataset.replace('cifar', ''))

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

    def load_lsun(self):
        transforms = self.transform(True, True, True)
        dataset = datasets.LSUN(self.path, classes=self.classes, transform=transforms)
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
        dataset = datasets.CIFAR10(self.path, transform=transforms, download=True)
        return dataset
    
    def load_cifar100(self):
        transforms = self.transform(True, True, True)
        dataset = datasets.CIFAR100(self.path, transform=transforms, download=True)
        return dataset
    
    def load_folder(self):
        transforms = self.transform(True, True, False)
        dataset = datasets.ImageFolder(self.path+'/'+self.dataset, transform=transforms)
        return dataset

    def loader(self):
        if 'load_' + self.dataset in dir(self):
            dataset = getattr(self, 'load_' + self.dataset)()
        else:
            dataset = self.load_folder()
        print('dataset',len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuffle,
                                              num_workers=4,
                                              drop_last=True)
        return dataloader
    
    @classmethod
    def from_dict(cls, params):
        return cls(params['dataset'], params['data_dir'], params['img_size'], params['batch_size'])