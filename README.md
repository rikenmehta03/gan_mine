# Exploring GAN: Improving training techniques

## Environment setup & dependency installation
```
git clone https://github.com/rikenmehta03/gan_mine.git
cd gan_mine
./install 
```
This commands will setup a virtual environment using python3 and install the required packages in the same environment. Install script will also create an alias for activating the virtual environment: `cv_env`

## Repository structure
Every module will be a directory containing `__init__.py` file and other subfiles. Below are the initial modules we need to write. 
- **gan** : This module contains trainer classes including generic gan trainer and BigGAN trainer. 
- **utils** : This module contains utility functions or classes we write. For example `Logger` class in `logger.py` file.
- **model** : This module contain all the different architecure we tried for discriminator or generator. 
- **data_loader** : This module provides wrapper for data-loader class for different datasets.
- **evaluation_metric** : Utility function and evaluation class used to evaluate various results.
- **scripts** : Ad-hoc scripts written to perform various experiments 

## Training script
Run these commands to run the training scripts
```
python main.py --model dcgan --dataset imagenet --data_path dataset/ --batch_size 64 --iters 100000 --log_iter 1000 --sn true --device cuda #train with spactral normalization
python main.py --dataset imagenet --data_path /var/www/dataset --batch_size 64 --iters 100000 --log_iter 1000 --sn true --device cuda --gpus 0,1,2,3 #Train with data parallel
```

## Evaluation script
Training process logs the sample images every 500 iterations. Model is saved according to the parameter `log_iter`. The results are saved in logs folder. To run the evaluation on the results, run the command 
```
evaluate.py --dataset imagenet --eval_folder imagenet_dcgan_sn_64_bs_64_18_12_2018_16:59:50 --device cuda --data_path dataset/ 
```

All other configurable parameters are available in `parameters.py` file.

## Results
- DCGAN on celeb-HQ

![celebhq](/samples/celebhq.jpeg)

- DCGAN on ImageNet*

![imagenet](/samples/imagenet.jpeg)

- BigGAN on LSUN (Church outdoor & Bridge)

![lsun](/samples/lsun.jpeg)