import json
import numpy as np
import glob

import matplotlib.pyplot as plt

def get_xy(summary, file):
    itrs = [x[0] for x in summary['inception_score']]
    scores = np.array([x[1][0] for x in summary['inception_score']])
    return itrs, scores

def get_fid(summary, file):
    itrs = [x[0] for x in summary['fid']]
    scores = np.array([x[1] for x in summary['fid']])
    return itrs, scores
    
        
def plot_graph(files, dataset, bs, get_xy):
    with open(files[0], 'r') as fp:
        summary = json.load(fp)
        _dir = files[0].split('/')[-2]
        x, scores_ = get_xy(summary, _dir)
        
    with open(files[1], 'r') as fp:
        summary = json.load(fp)
        _dir = files[1].split('/')[-2]
        x, scores_sn = get_xy(summary, _dir)
    
    line_ = plt.plot(x, scores_, 'b', label='DCGAN')
    line_sn = plt.plot(x, scores_sn, 'r', label='SN-DCGAN')
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Inception Score')
    print("Dataset: ", dataset)
    print("Batch Size: ", bs)
    plt.show()
    
datasets = ['imagenet', 'cifar10', 'celebhq', 'celeb']
log_folder = 'gan_mine/logs/'

for _d in datasets:
    for bs in [32, 64]:
        folder_ = glob.glob(os.path.join(log_folder, '{}_dcgan_64_bs_{}_*'.format(_d, bs), 'summary.json'))[0]
        folder_sn = glob.glob(os.path.join(log_folder, '{}_dcgan_sn_64_bs_{}_*'.format(_d, bs), 'summary.json'))[0]
        plot_graph((folder_, folder_sn), _d, bs, get_xy)
        plot_graph((folder_, folder_sn), _d, bs, get_fid)
        
        print("Printing IS and FID scores")
        with open(folder_, 'r') as fp:
            summary = json.load(fp)
            
        print("dataset: ", _d, "batch_size: ", bs)
        print("Without SN:")
        print("IS: ", summary['best_inception'][1][0], summary['best_inception'][1][1])
        print("FID: ", summary['best_fid'][1])
        with open(folder_sn, 'r') as fp:
            summary = json.load(fp)
        print("With SN:")
        print("IS: ", summary['best_inception'][1][0],summary['best_inception'][1][1])
        print("FID: ", summary['best_fid'][1])