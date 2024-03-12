import torch as pt
import numpy as np
import matplotlib

from skimage import data

# filter 
from smoothing.src import L0_Smoothing

def activate(x):
    return 2 * np.tanh(x)

def feature_weighted_avg(y_grad):
    '''
    Equation 5, used for computing the loss function for first component of model
    '''
    return sum()

def cost_a(y_grad, ):
    '''
    Cost function for first stage of network
    '''
    return

def cost_b():
    '''
    Cost function for first stage of network
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