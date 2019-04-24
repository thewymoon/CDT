import numpy as np
from scipy.stats import multivariate_normal, entropy
import scipy
from scipy.special import expit
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import torch


### FUNCTION DEFINITIONS ####

def flip_beta(beta):
    flipped_beta = np.zeros(len(beta))
    for i in range(int(len(beta)/4)):
        flipped_beta[4*i:(4*i+4)] = beta[4*i:(4*i+4)][::-1]                                         
    return flipped_beta

def faster_dot(X_matrices, beta):
    result = np.empty((X_matrices.shape[0],X_matrices.shape[1]))
    for i in range(len(X_matrices)):
        np.dot(X_matrices[i], beta, result[i])
    return result

def better_return_counts_weighted(labels, classifications, classes, weights):
    output = []
    
    class_indices = []
    for class_value in classes:
        class_indices.append(np.where(labels==class_value)[0])

    #indices_first_class = np.where(labels == classes[0])[0]
    #indices_second_class = np.where(labels == classes[1])[0]
    
    for c in classifications:
        temp = []
        for i in range(2):
            temp.extend([np.sum(weights[np.where(c[indices]==i)[0]]) for indices in class_indices])

        output.append(temp)
        #output.append([np.sum(weights[np.where(c[indices_first_class] == 0)[0]]),
        #              np.sum(weights[np.where(c[indices_second_class] == 0)[0]]),
        #              np.sum(weights[np.where(c[indices_second_class] == 1)[0]]),
        #              np.sum(weights[np.where(c[indices_first_class] == 1)[0]])])
    return np.array(output)

def calculate_prob(N, l, x):
    return 1 - (scipy.special.bdtr(x-1, l, (1/4))**(2*(N-l+1)))

def find_best_div(N, l, proportion):
    best = -1
    closest = 1
    for d in np.arange(1,l+1):
        diff = np.abs((calculate_prob(N, l, d) - proportion))
        if diff < closest:
            closest = diff
            best = d
        else:
            pass
    return best

def x_to_string(x):
    return "".join([str(i) for i in x])

def x_to_matrix(x, motif_length):
    return np.vstack([x[i:i+motif_length*4] for i in range(0,len(x)-(motif_length-1)*4,4)])

def motif_to_beta(motif):
    A = [1.0,0.0,0.0,0.0]
    C = [0.0,1.0,0.0,0.0]
    G = [0.0,0.0,1.0,0.0]
    T = [0.0,0.0,0.0,1.0]
    convertdict = {'A':A, 'C':C, 'G':G, 'T':T}

    return np.array([convertdict[x] for x in motif]).flatten()

def H_hat(pos, neg):
    return -(pos/(pos+neg))*np.log(pos/(pos+neg)) - (neg/(pos+neg))*np.log(neg/(pos+neg))
def IG(P, N, p, n):
    parent_entropy = H_hat(P,N)
    children_entropy = ((p+n)/(P+N))*H_hat(p,n) + ((P+N-p-n)/(P+N))*H_hat(P-p, N-n)

    return parent_entropy - children_entropy


def convDNA_single(X, X_rc, B):
    X_size = X.size(0)
    #result = np.empty((0,X_size), dtype=np.float64)

    conv = torch.nn.Conv1d(1,1,kernel_size=len(B),stride=4,bias=False)
    conv.weight.data = torch.from_numpy(B).float().reshape(1,1,-1)
    if torch.cuda.is_available():
        conv = conv.cuda()

    forward = conv(X).cpu().data.numpy().squeeze(1)
    reverse = conv(X_rc).cpu().data.numpy().squeeze(1)

    return forward, reverse

def convDNA_single_maxinfo(X, X_rc, B):
    X_size = X.size(0)
    forward, reverse = convDNA_single(X, X_rc, B)

    forward_max = np.max(forward, axis=1)
    reverse_max = np.max(reverse, axis=1)

    max_vals = np.max((forward_max, reverse_max), axis=0)
    max_pos = np.argmax((forward_max, reverse_max), axis=0)
    #max_pos = np.argmax((forward, reverse), axis = 0)
    forward_max_pos = np.argmax(forward, axis=1)
    reverse_max_pos = np.argmax(reverse, axis=1)
    max_strand = np.argmax((forward_max, reverse_max), axis=0) ### 0 for forward and 1 for reverse


    X_cpu = X.data.cpu().numpy().squeeze(1)
    X_rc_cpu = X_rc.data.cpu().numpy().squeeze(1)
    
    #max_sequences = np.array([X_cpu[i][int(4*max_pos[i]):int(4*(max_pos[i])+len(B))] if max_strand[i]==0 else X_rc_cpu[i][int(4*max_pos[i]):int(4*(max_pos[i])+len(B))] for i in range(X_size)])
    max_sequences = np.array([X_cpu[i][int(4*forward_max_pos[i]):int(4*(forward_max_pos[i])+len(B))] if max_strand[i]==0 else X_rc_cpu[i][int(4*reverse_max_pos[i]):int(4*(reverse_max_pos[i])+len(B))] for i in range(X_size)])

    return max_vals, max_sequences


def threshold_DNA(X, X_rc, B, conv):
    output1, output2 = convDNA_single(X, X_rc, B, conv)

    return 0
    

def pytorch_convDNA(X, X_rc, B, conv, threshold, single=False, limit=2000):
    X_size = X.size(0)
    result = np.empty((0,X_size), dtype=np.float64)
    for i in range(int(len(B)/limit)):
        conv.weight.data = torch.from_numpy(B[i*limit:(i+1)*limit].reshape(limit,1,len(B[0]))).float()
        if torch.cuda.is_available():
            conv = conv.cuda()
            
        output1 = conv(X)
        output2 = conv(X_rc)
        max_output = np.swapaxes((torch.max(output1, output2).max(dim=2)[0] >= threshold).cpu().data.numpy(),0,1)
        result = np.append(result, max_output, axis=0)
    return result

def pytorch_convDNA_max(X, X_rc, B, conv, single=False, limit=2000):
    X_size = X.size(0)
    result = np.empty((0,X_size), dtype=np.float64)
    for i in range(int(len(B)/limit)):
        conv.weight.data = torch.from_numpy(B[i*limit:(i+1)*limit].reshape(limit,1,len(B[0]))).float()
        if torch.cuda.is_available():
            conv.cuda()
            
        output1 = conv(X)
        output2 = conv(X_rc)

        torch_max = torch.max(output1, output2).max(dim=2)
        max_output = np.swapaxes((torch_max[0]).cpu().data.numpy(),0,1)
        result = np.append(result, max_output, axis=0)
    return result

def pytorch_conv2d(X, B, conv, threshold=3000, limit=100):
    X_size = X.size(0)
    result = np.empty((0,X_size), dtype=np.float64)
    for i in range(int(len(B)/limit)):
        conv.weight.data = torch.from_numpy(B[i*limit:(i+1)*limit].reshape(limit,1,B.shape[1],B.shape[2])).float()
        conv = conv.cuda()
        output = np.swapaxes((conv(X).max(dim=2)[0].max(dim=2)[0] >= threshold).cpu().data.numpy(),0,1)
        result = np.append(result, output, axis=0)
    return result

def pytorch_conv_exact2d(X, B, conv, threshold=500, limit=50):
    X_size = X.size(0)
    X=X.cuda()
    result = np.empty((0,X_size), dtype=np.float64)
    conv.out_channels = limit
    for i in range(int(len(B)/limit)):
        conv.weight.data = torch.from_numpy(B[i*limit:(i+1)*limit].reshape(limit,1,B.shape[1],B.shape[2])).float()
        conv = conv.cuda()
        output = np.swapaxes((conv(X).max(dim=2)[0].max(dim=2)[0] >= threshold).cpu().data.numpy(),0,1)
        result = np.append(result, output, axis=0)

    leftover = len(B) % limit
    conv.out_channels = leftover
    conv.weight.data = torch.from_numpy(B[(-1)*leftover:].reshape(leftover,1,B.shape[1],B.shape[2])).float()
    conv = conv.cuda()
    output = np.swapaxes((conv(X).max(dim=2)[0].max(dim=2)[0] >= threshold).cpu().data.numpy(),0,1)
    result = np.append(result, output, axis=0)

    return result



def pytorch_convDNA_single(X, X_rc, B, conv, threshold):
    conv.weight.data = torch.from_numpy(B).float()
    if torch.cuda.is_available():
        conv = conv.cuda()
    output_forward = conv(X)
    output_rc =conv(X_rc)
    classifications = np.swapaxes((torch.max(output_forward, output_rc).max(dim=2)[0] >= threshold).cpu().data.numpy(),0,1)
    return classifications


def pytorch_conv2d_single(X, B, conv, threshold):
    conv.weight.data = torch.from_numpy(B).float()
    if torch.cuda.is_available():
        conv = conv.cuda()
    output_forward = conv(X)
    classifications = np.swapaxes((output_forward.max(dim=2)[0].max(dim=2)[0] >= threshold).cpu().data.numpy(),0,1)
    return classifications

