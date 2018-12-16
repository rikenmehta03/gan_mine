import argparse

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='dcgan', choices=['dcgam', 'biggan'])
    parser.add_argument('--sn', type= lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--sa', type= lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--img_size', type=int, default=64)

    # Training setting
    parser.add_argument('--iters', type=int, default=25000, help='number of iterations to train')
    parser.add_argument('--log_iter', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--g_lr', type=float, default=0.0002)
    parser.add_argument('--d_lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # using pretrained
    parser.add_argument('--resume', type=str, default=None)

    # Misc
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpus', type=str, default='0', help='gpuids eg: 0,1,2,3')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'celeb', 'celebhq', 'imagenet', 'lsun'])
    parser.add_argument('--classes', type=str, default=None, help='provide classes for lsun dataset. eg: bedroom_train,church_outdoor_train,bridge_train')

    # Path
    parser.add_argument('--data_path', type=str, default='../dataset')
    parser.add_argument('--index_path', type=str, default='../indices')
    parser.add_argument('--eval_folder', type=str)

    return parser.parse_args()
