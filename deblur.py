import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

from skimage import data

# L_0 filter. I did not implement this. 
from smoothing.src import L0_Smoothing

## model params
filter_count = 128 # 128 filters per layer

filter_size = {
    1: 9,
    2: 1,
    3: 3,
    4: 5,
    5: 1,
    6: 3
}
train_lam = 0.005
train_eps = 10e-6

ker_eta = 0.002
ker_gam = 1
ker_tau = 15
ker_bet = 0.001

class Net(nn.Module):
    def __init__(self):
        super(Net, self).init()


def re_lu(x):
    '''rectified linear unit, used in equation 5 from source'''
    return max( 0, x)

def activate(x):
    '''activation function, used in equation 4 from source'''
    return 2 * np.tanh(x)

def compute_feature(_feature, _filter, _bias):
    '''computes a single feature, equation 4 from source'''
    output = sum([np.convolve(_feature[m], w) + _bias for m,w in enumerate(_filter)])
    return re_lu(output)

def feature_weighted_avg(_feature, _filter, _bias):
    '''
    Equation 5, used for computing the loss function for first component of model.
    Applied only to final layer of the model. 
    '''
    output = sum([np.convolve(_feature[m], w) + _bias for m,w in enumerate(_filter)])
    return activate(output)

def cost_a(y_grad, ):
    '''
    Cost function for first stage of network
    '''
    return

def cost_b():
    '''
    Cost function for second stage of network
    '''
    return

def img_grad(image) -> np.array:
    '''
    Given input image, returns gradient-space representation of the image
    '''
    return

Weights = []


# algorithm 1 - training the proposed network

# input: training dataset

# first stage
# for each image
## compute output of first stage with 2,3,5
## compute loss between O_1 and T_1 with 7 and update network weights with backprop

# second stage
# for each image 
## compute the output of first stage with 2,3,5
## compute the output of the second stage with 8,9,10
## compute loss between O_2 and T_2 with 11, update using backprop

# combine the subnetworks using 12

# fine tuning
# for each image patch
## compute the output of the whole network with 2,3,4
## compute loss between f_w(dy_i) and T_2(x_i), update using backprop

# output: network model weights W

# Algorithm 2 - Blur kernel and latent image estimation

# input: blurred input y, network weights W

# compute salient edges de = f_W(dy) according to 2,3,4

# for loop = 1: tau 
## solve for k using 14
## solve for x_hat using 15
## de <- dx

# estimate latent image x using 17

# output: blur kernel k and latent image x