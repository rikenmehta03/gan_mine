import os
import json
import torch
import numpy as np
from scipy import linalg
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d

from .inception import InceptionV3

curr_dir = os.path.dirname(__file__)

def get_activations(dataloader, model, dims, device, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- dataloader  : pytorch dataloader for images
    -- model       : Instance of inception model
    
    -- dims        : Dimensionality of features returned by Inception
    -- device      : pytorch device object
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    
    data_iter = dataloader
    n_batches = len(data_iter)
    batch_size = dataloader.batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, dims))

    for i, (batch, _) in enumerate(data_iter):
        
        start = i * batch_size
        end = start + batch_size
        batch = batch.to(device)
        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an 
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an 
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(dataloader, model, dims, device, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- dataloader  : pytorch dataloader
    -- model       : Instance of inception model
    -- dims        : Dimensionality of features returned by Inception
    -- device      : pytorch device object
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(dataloader, model, dims, device, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def fid_score(s_dataloader, s_name, g_dataloader, dims=2048, device = torch.device('cpu')):
    """Calculation of FID.
    Params:
    -- s_dataloader  : pytorch dataloader for real images
    -- g_dataloader  : pytorch dataloader for generated images
    -- dims          : Dimensionality of features returned by Inception
    -- device        : pytorch device object
    
    Returns:
    -- fid_value : FID
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(g_dataloader, model, dims, device)

    if os.path.exists(os.path.join(curr_dir, 'mu_sigma.npy')):
        mu_sigma = np.load(os.path.join(curr_dir, 'mu_sigma.npy'))[()]
    else:
        mu_sigma = {}
    
    if s_name in mu_sigma:
        _m_s = mu_sigma[s_name]
        m2, s2 = _m_s[0], _m_s[1]
    else:
        m2, s2 = calculate_activation_statistics(s_dataloader, model, dims, device)
        if os.path.exists(os.path.join(curr_dir, 'mu_sigma.npy')):
            mu_sigma = np.load(os.path.join(curr_dir, 'mu_sigma.npy'))[()]
        mu_sigma[s_name] = [m2, s2]
        np.save(os.path.join(curr_dir, 'mu_sigma.npy'), mu_sigma)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value