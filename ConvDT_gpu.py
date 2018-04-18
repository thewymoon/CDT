import pandas as pd
import numpy as np
import copy 
import itertools
from numpy.lib.stride_tricks import as_strided
from scipy.stats import multivariate_normal
import scipy
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

import torch
import torch.nn as nn
from torch.autograd import Variable

### FUNCTION DEFINITIONS ####

# Loss Function
def entropy(p_vec, pseudo=0.00000001):
    if np.sum(p_vec) > 0:
        return np.sum([-(p)*np.log((p)) for p in [(x/np.sum(p_vec))+pseudo for x in p_vec]])
    else:
        return 0

def two_class_weighted_entropy(counts, pseudo=.01):
    return (entropy([counts[0], counts[1]], pseudo=pseudo)*np.sum(counts[0:2]) + entropy([counts[2], counts[3]], pseudo=pseudo)*np.sum(counts[2:4]))/np.sum(counts)

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

# CLASSIFY SEQUENCES

def classify_sequences(X_matrices, X_matrices_rc, beta):
    return np.logical_or(np.any(faster_dot(X_matrices, beta) >= 1.0, axis=1), np.any(faster_dot(X_matrices_rc, beta) >= 1.0, axis=1)).astype(int)

def weighted_classify_sequences(X_matrices, X_matrices_rc, beta, weights):
    return np.logical_or(np.any(faster_dot(X_matrices, beta) >= 1.0, axis=1), np.any(faster_dot(X_matrices_rc, beta) >= 1.0, axis=1)).astype(int)
    
#def classify_sequence(x, beta, motif_length):
#    x_matrix = x_to_matrix(x, motif_length) 
#    print(x_matrix.shape)
#    return int(np.any(np.dot(x_matrix, beta) >= 1.0) or np.any(np.dot(x_matrix, flip_beta(beta)) >= 1.0))

def classify_sequence(x, beta, motif_length):
    flipped_beta = beta[::-1]

    scan_length = int((len(x) - len(beta))/4 + 1)
    #print('scan length', scan_length, len(x), len(beta))

    out1 = np.array([np.dot(x[(4*i):(4*i)+len(beta)], beta) for i in range(scan_length)])
    out2 = np.array([np.dot(x[(4*i):(4*i)+len(beta)], flipped_beta) for i in range(scan_length)])

    return int(np.any((out1>1.0) + (out2>1.0)))


    


###########################
#### COUNT FUNCTIONS ######
###########################

# returns how many true pos, false, pos, true neg, false neg
def return_counts(labels, classifications):
    zipped = list(zip(labels, classifications))
    true1 = zipped.count((1,1))
    false1 = zipped.count((0,1))
    true0 = zipped.count((0,0))
    false0 = zipped.count((1,0))

    return [true1, false1, true0, false0]

def return_counts_general(labels, classifications, classes): ## use the collection.Counter() method to make this faster!!
    zipped = list(zip(labels, classifications))
    true_first_class = zipped.count((classes[0], classes[0]))
    false_first_class = zipped.count((classes[1], classes[0]))
    true_second_class = zipped.count((classes[1], classes[1]))
    false_second_class = zipped.count((classes[0], classes[1]))

    return [true_first_class, false_first_class, true_second_class, false_second_class]

def find(lst, elements):
    return [[i for i,x in enumerate(lst) if x==e] for e in elements]

def return_counts_weighted(labels, classifications, classes, weights):
    zipped = list(zip(labels, classifications))

    true_first_class, false_first_class, true_second_class, false_second_class = find(zipped, [(classes[0],classes[0]), (classes[1],classes[0]), (classes[1],classes[1]), (classes[0],classes[1])])

    return [np.sum(weights[true_first_class]), np.sum(weights[false_first_class]), np.sum(weights[true_second_class]), np.sum(weights[false_second_class])]

def better_return_counts_weighted(labels, classifications, classes, weights):
    output = []
    indices_first_class = np.where(labels == classes[0])[0]
    indices_second_class = np.where(labels == classes[1])[0]
    
    for c in classifications:
        output.append([np.sum(weights[np.where(c[indices_first_class] == classes[0])[0]]),
                      np.sum(weights[np.where(c[indices_second_class] == classes[0])[0]]),
                      np.sum(weights[np.where(c[indices_second_class] == classes[1])[0]]),
                      np.sum(weights[np.where(c[indices_first_class] == classes[1])[0]])])
    return np.array(output)

def return_weightedcounts(labels, classifications, weights):
    zipped = list(zip(list(zip(labels, classifications)), weights))
    true1 = np.sum([a[1] for a in zipped if a[0]==(1,1)])
    false1 = np.sum([a[1] for a in zipped if a[0]==(0,1)]) 
    true0 = np.sum([a[1] for a in zipped if a[0]==(0,0)])
    false0 = np.sum([a[1] for a in zipped if a[0]==(1,0)])
    return [true1, false1, true0, false0]



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


#def x_to_matrix(x, motif_length):
#    numpy_arrayx = np.array(x)
#    size = numpy_arrayx.itemsize
#    sequence_length = int(len(x)/4)
#
#    #print('size', size)
#    return as_strided(numpy_arrayx, shape = [sequence_length - motif_length, motif_length*4], strides = [size*4,size])

def x_to_matrix(x, motif_length):
    return np.vstack([x[i:i+motif_length*4] for i in range(0,len(x)-(motif_length-1)*4,4)])


def motif_to_beta(motif):
    A = [1.0,0.0,0.0,0.0]
    C = [0.0,1.0,0.0,0.0]
    G = [0.0,0.0,1.0,0.0]
    T = [0.0,0.0,0.0,1.0]
    convertdict = {'A':A, 'C':C, 'G':G, 'T':T}

    return np.array([convertdict[x] for x in motif]).flatten()

def normalize_dict(d):
    d_copy = copy.deepcopy(d)
    total = sum(d.values())
    for k in d_copy:
        d_copy[k] /= total
    return d_copy


def _get_member_scores(X_matrices, X_matrices_rc, y, classes, weights, m):
            return two_class_weighted_entropy(return_counts_weighted(y, classify_sequences(X_matrices, X_matrices_rc, m), classes, weights))


def pytorch_conv(X, X_rc, B, conv, single=False, limit=2000):
    X_size = X.size(0)
    if single:
        result = np.empty((0,X_size), dtype=np.float64)
        for i in range(int(len(B)/limit)):
            conv.weight.data = torch.from_numpy(B.reshape(1,1,len(B))).float()
            if torch.cuda.is_available():
                conv.cuda()

            output1 = (conv(X) >= 1.0).sum(dim=2)
            output2 = (conv(X_rc) >= 1.0).sum(dim=2)

            result = np.append(result, np.swapaxes((output1+output2).cpu().data.numpy().astype(bool),0,1), axis=0)

    else:
        result = np.empty((0,X_size), dtype=np.float64)
        for i in range(int(len(B)/limit)):
            conv.weight.data = torch.from_numpy(B[i*limit:(i+1)*limit].reshape(limit,1,len(B[0]))).float()
            if torch.cuda.is_available():
                conv.cuda()
                
            output1 = conv(X)
            output2 = conv(X_rc)
            max_output = np.swapaxes((torch.max(output1, output2).max(dim=2)[0] > 1.0).cpu().data.numpy(),0,1)
            result = np.append(result, max_output, axis=0)

    return result

    

#########################
### CLASS DEFINITIONS ###
#########################

from sklearn.base import BaseEstimator, ClassifierMixin
class ConvDT(BaseEstimator):
    def __init__(self, depth, motif_length, sequence_length, iterations=10, num_processes=4, alpha=0.80, loss_function=two_class_weighted_entropy, optimization_sample_size=(2000,2000)):
        self.depth = depth
        self.motif_length = motif_length
        self.sequence_length = sequence_length
        self.iterations = iterations
        self.alpha = alpha
        self.num_processes = num_processes
        self.loss_function = loss_function
        self.data = []
        self.optimization_sample_size = optimization_sample_size
        self.conv = nn.Conv1d(1,1000,kernel_size=motif_length*4,stride=4,bias=False)
        self.conv_single = nn.Conv1d(1,1,kernel_size=motif_length*4,stride=4,bias=False)
        self.conv_betas = nn.Conv1d(1,1,kernel_size=motif_length*4,stride=4,bias=False)
                

    def _find_optimal_beta(self, X, X_rc, indices, y, weights, grid, cov_init=0.4, elite_num=20):

        cov = cov_init
        best_memory = None
        best_score = 999
        best_classifications = None
        
        ### sample members (betas) ###
        for i in range(self.iterations):
            print('iteration:', i)
            if i==0:
                members = grid[np.random.choice(range(len(grid)), size=self.optimization_sample_size[0], replace=False)]
            else:
                members = multivariate_normal.rvs(mean=mu, cov=cov, size=self.optimization_sample_size[1])

            print('calculating scores...')
            
            ####################
            ### PYTORCH PART ###
            ####################

            #### TESTING ####
            indices_cuda = Variable(torch.LongTensor(indices))
            if torch.cuda.is_available():
                indices_cuda = indices_cuda.cuda()
            classifications = pytorch_conv(X.index_select(dim=0, index=indices_cuda), 
                                           X_rc.index_select(dim=0, index=indices_cuda), members, self.conv, limit=1000)
            #brute_classifications = np.array([classify_sequence(x[0], members[0], self.motif_length) for x in X.index_select(dim=0, index=indices_cuda).cpu().data.numpy()])
            #print(classifications.shape)
            #print(brute_classifications.shape)
            #print('total classifications', len(classifications))
            #print('matches', np.count_nonzero(classifications[0] == brute_classifications))
            #print('classifications', np.sum(classifications[0]))
            #print('brute class', np.sum(brute_classifications))

            # get the entropy scores for each member (beta)
            member_scores = np.apply_along_axis(two_class_weighted_entropy,1,better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices]))

            #print('getting best scores')
            best_scoring_indices = np.argsort(member_scores)[0:elite_num]
            if member_scores[best_scoring_indices[0]] < best_score:
                best_score = member_scores[best_scoring_indices[0]]
                best_memory = members[best_scoring_indices[0]]
                best_classifications = classifications[best_scoring_indices[0]]
            else:
                pass
            
            print('best score so far:', best_score)

            ## Calculate the MLE ##
            new_mu = np.mean(members[best_scoring_indices], axis=0)
            new_cov = np.mean([np.outer(x,x) for x in (members[best_scoring_indices] - new_mu)], axis=0) ## maybe faster way

            if i==0:
                mu = new_mu
                cov = self.alpha*new_cov + (1-self.alpha)*cov
            else:
                mu = self.alpha*new_mu + (1-self.alpha)*mu
                cov = self.alpha*new_cov + (1-self.alpha)*cov

        self.conv_single.weight.data = torch.from_numpy(mu.reshape(1,1,self.motif_length*4)).float()
        if torch.cuda.is_available():
            self.conv_single = self.conv_single.cuda()
        output_forward = self.conv_single(X.index_select(dim=0, index=indices_cuda))
        output_rc = self.conv_single(X_rc.index_select(dim=0, index=indices_cuda))
        classifications = np.swapaxes((torch.max(output_forward, output_rc).max(dim=2)[0] >= 1.0).cpu().data.numpy(),0,1)

        if self.loss_function(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices])[0]) > best_score:
            print("going with something else")
            beta = best_memory
            output_classifications = best_classifications
        else:
            print("we good")
            beta = mu
            output_classifications = classifications[0]

        return beta, (indices[np.where(output_classifications==1)[0]], indices[np.where(output_classifications==0)[0]])

    def fit(self, X, y, sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones(len(X))

        self.data = []
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.betas = []
        self.proportions = []

        X_gpu = Variable(torch.from_numpy(X.reshape(X.shape[0], 1, X.shape[1])).float())
        Xrc_gpu = Variable(torch.from_numpy(np.array([x[::-1] for x in X]).reshape(X.shape[0], 1, X.shape[1])).float())
        if torch.cuda.is_available():
            X_gpu = X_gpu.cuda()
            Xrc_gpu = Xrc_gpu.cuda()

        print('creating grid')
        nucleotides = ['A', 'C', 'G', 'T']
        keywords = itertools.product(nucleotides, repeat=self.motif_length)
        kmer_list = ["".join(x) for x in keywords]
        div = find_best_div(self.sequence_length, self.motif_length, 0.5)
        full_grid = np.array([motif_to_beta(x) for x in kmer_list]) / div

        for layer in range(self.depth):
            if layer == 0:
                b, splits = self._find_optimal_beta(X_gpu, Xrc_gpu, np.arange(len(X)), y, sample_weight, full_grid)
                print("splits", len(splits[0]), len(splits[-1]))
                self.betas.append([b])
                self.data.append([splits])

            else:
                for i in range(len(self.betas[layer-1])):
                    left = self.data[layer-1][i][0]
                    right = self.data[layer-1][i][1]

                    left_beta, left_children = self._find_optimal_beta(X_gpu, Xrc_gpu, left, y, sample_weight, full_grid)
                    print("going left", len(left_children[0]), len(left_children[1]))
                    
                    right_beta, right_children = self._find_optimal_beta(X_gpu, Xrc_gpu, right, y, sample_weight, full_grid)
                    print("going right", len(right_children[0]), len(right_children[1]))

                    if i==0: #have to append instead of extend on first iteration
                        self.betas.append([left_beta, right_beta])
                        self.data.append([left_children, right_children])
                    else:
                        self.betas[layer].extend([left_beta, right_beta])
                        self.data[layer].extend([left_children, right_children])

        for i in range(len(self.betas[-1])):
            left = self.data[-1][i][0]
            right = self.data[-1][i][1]
            print("LEFT", len(left))
            print("RIGHT", len(right))
            left_proportion = np.average(y.take(left) == self.classes_[0], weights=sample_weight.take(left), axis=0)
            right_proportion = np.average(y.take(right) == self.classes_[0], weights=sample_weight.take(right), axis=0)
            self.proportions.extend([(left_proportion, 1-left_proportion), (right_proportion, 1-right_proportion)])

        print(self.proportions)
        return self
        

    def predict_proba_one(self, x):
        current_layer = 0
        position = 0
        for current_layer in range(self.depth):
            out = classify_sequence(x, self.betas[current_layer][position], self.motif_length)
            if out==1:#self.classes_[0]: ## if motif found, go to left node
                position = position*2
            else:
                position = position*2+1
                
        return self.proportions[position]

    #def predict_proba_old(self, X):
    #    return np.array([self.predict_proba_one(x) for x in X])

    def predict_proba(self, X):
        X_pytorch = Variable(torch.from_numpy(X.reshape(X.shape[0], 1, X.shape[1])).float())
        Xrc_pytorch = Variable(torch.from_numpy(np.array([x[::-1] for x in X]).reshape(X.shape[0], 1, X.shape[1])).float())
        
        flattened_betas = np.array(list(itertools.chain(*self.betas)))
        self.conv_betas.out_channels = len(flattened_betas)
        
        self.conv_betas.weight.data = torch.from_numpy(flattened_betas.reshape(len(flattened_betas),1,len(flattened_betas[0]))).float()
        if torch.cuda.is_available():
            self.conv_betas.cuda()
            X_pytorch = X_pytorch.cuda()
            Xrc_pytorch = Xrc_pytorch.cuda()
        output1 = self.conv_betas(X_pytorch)
        output2 = self.conv_betas(Xrc_pytorch)
        betas_output = np.swapaxes((torch.max(output1, output2).max(dim=2)[0] > 1.0).cpu().data.numpy(),0,1)
        
        output = []
        offset = 2**(self.depth) - 1
        for i in range(len(X)):
            idx = 0
            for current_layer in range(self.depth):
                if betas_output[idx, i]:
                    idx = idx*2 + 1
                else:
                    idx = idx*2 + 2
                
            output.append(self.proportions[idx - offset])
        
        return np.array(output)
        
    def predict(self, X):
        output = []
        thresh = 0.5
        predicted_probs = self.predict_proba(X)
        for o in predicted_probs:
            if o[0] > thresh:
                output.append(self.classes_[0])
            else:
                output.append(self.classes_[1])

        return output


    def score(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X))
            
        #return np.sum(self.predict(X) == y) / len(y)
        return np.average(self.predict(X) == y, weights=sample_weight, axis=0)






################################################################################################################


def print_with_features(L, Features, ordered=False):
    if ordered==False:
        for i in range(len(features)):
            print(L[i], Features[i])
    else:
        print_with_features([L[x] for x in np.argsort(L)[::-1]],
                [Features[x] for x in np.argsort(L)[::-1]])

def update_importances(importances, tree, weights, alpha):
    if len(importances) != len(weights):
                raise ValueError
    else:
        for i in range(len(importances)):
            importances[i] += tree.feature_importances_ * weights[i] * alpha

def normalize(x):
    return x/np.sum(x)

def predict_proba_importances(X, BDTLIST):
    output = []
    for b in BDTLIST:
        output.append(b.predict(X.reshape(1,-1))[0])

    return output




def plot_roc(true_y, proba_y):
    plt.figure(figsize=(8,5))
    false_pos, true_pos, _ = roc_curve(true_y, proba_y)
    roc_auc = auc(false_pos, true_pos)

    plt.plot(false_pos, true_pos)
    plt.text(.6,.1,"AUC: " + str("%.4f" % roc_auc), fontsize=20)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")


def plot_motif(motif, size=(8,5)):
    motif_length = int(len(motif)/4)
    optimal_beta = np.array([motif[4*i:4*i+4] for i in range(motif_length)])


    plt.figure(figsize=size)
    width = 0.2
    plt.bar(left = np.arange(motif_length), height=optimal_beta[:,0], width=width, color='r', label="A")
    plt.bar(left = np.arange(motif_length)+width, height=optimal_beta[:,1], width=width, color='b', label='C')
    plt.bar(left = np.arange(motif_length)+width*2, height=optimal_beta[:,2], width=width, color='m', label='G')
    plt.bar(left = np.arange(motif_length)+width*3, height=optimal_beta[:,3], width=width, color='g', label='T')
    plt.xlabel("Position")

    plt.legend(bbox_to_anchor=(1.1, 1.05))











