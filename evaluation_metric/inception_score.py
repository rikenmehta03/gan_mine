import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(dataloader, device, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    dataloader -- Torch dataloader for the images
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    
    assert batch_size > 0
    assert N >= batch_size

    # Set up dtype
    # if cuda:
    #     dtype = torch.cuda.FloatTensor
    # else:
    #     if torch.cuda.is_available():
    #         print("WARNING: You have a CUDA device, so you should probably set cuda=True")
    #     dtype = torch.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, (batch, _) in enumerate(dataloader, 0):
        batch = batch.to(device)
        batchv = Variable(batch).to(device)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)