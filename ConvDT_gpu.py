import pandas as pd
import numpy as np
import copy 
import itertools
from scipy.stats import multivariate_normal, entropy
import scipy
from scipy.special import expit
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

from ConvFunctions import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.base import BaseEstimator, ClassifierMixin


#########################
### CLASS DEFINITIONS ###
#########################

class ConvDTClassifier2d(BaseEstimator):
    def __init__(self, depth, kernel_shape=6, iterations=12, threshold=500, num_processes=4, alpha=0.80, loss_function=multi_class_weighted_entropy, optimization_sample_size=(2000,2000), filters_limit=100, batch_size=None, min_node_frac=0.01, CE_elite_num=20, CE_cov_init=0.4, image_size=(28,28)):
        self.depth = depth
        self.kernel_shape = kernel_shape
        self.iterations = iterations
        self.alpha = alpha
        self.num_processes = num_processes
        self.loss_function = loss_function
        self.threshold = threshold
        self.filters_limit = filters_limit
        self.min_node_frac = min_node_frac
        self.data = []
        self.optimization_sample_size = optimization_sample_size
        self.CE_elite_num = CE_elite_num
        self.CE_cov_init = CE_cov_init
        self.conv = nn.Conv2d(1,filters_limit,kernel_size=kernel_shape,bias=False)
        self.conv_single = nn.Conv2d(1,1,kernel_size=kernel_shape,bias=False)
        self.conv_betas = nn.Conv2d(1,1,kernel_size=kernel_shape,bias=False)
        self.image_size=image_size
                

    def _find_optimal_beta(self, X, indices, y, weights, grid):

        if len(indices) < self.min_node_frac * len(X):
            print("Not enough samples: SKIPPING")
            return np.zeros((self.kernel_shape, self.kernel_shape)), (indices, indices)

        cov = self.CE_cov_init
        best_memory = None
        best_score = 99999
        best_classifications = None
        
        ### sample members (betas) ###
        for i in range(self.iterations):
            print('iteration:', i)
            if i==0:
                members = grid[np.random.choice(range(len(grid)), size=self.optimization_sample_size[0], replace=False)]
            else:
                members = multivariate_normal.rvs(mean=mu.flatten(), cov=cov, size=self.optimization_sample_size[1]).reshape(self.optimization_sample_size[1],self.kernel_shape, self.kernel_shape)

            print('calculating scores...')
            
            ####################
            ### PYTORCH PART ###
            ####################

            #### TESTING ####
            indices_cuda = Variable(torch.LongTensor(indices))
            if torch.cuda.is_available():
                indices_cuda = indices_cuda.cuda()
            classifications = pytorch_conv2d(X.index_select(dim=0, index=indices_cuda), members, self.conv, threshold=self.threshold, limit=self.filters_limit)

            # get the entropy scores for each member (beta)
            member_scores = np.apply_along_axis(self.loss_function,1,better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices]))


            #print('getting best scores')
            best_scoring_indices = np.argsort(member_scores)[0:self.CE_elite_num]
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

        self.conv_single.weight.data = torch.from_numpy(mu.reshape(1,1,self.kernel_shape,self.kernel_shape)).float()
        if torch.cuda.is_available():
            self.conv_single = self.conv_single.cuda()
        output_forward = self.conv_single(X.index_select(dim=0, index=indices_cuda))
        classifications = np.swapaxes((output_forward.max(dim=2)[0].max(dim=2)[0] >= self.threshold).cpu().data.numpy(),0,1)
        print('RIGHT HERE', classifications.shape)

        if child_variance(y[indices], classifications)[0] > best_score:
            print("going with something else")
            beta = best_memory
            output_classifications = best_classifications
        else:
            print("we good")
            beta = mu
            output_classifications = classifications[0]

        going_left = np.where(output_classifications==1)[0]
        going_right = np.where(output_classifications==0)[0]
        
        if len(going_left)==0:
            going_left = going_right
        elif len(going_right)==0:
            going_right = going_left
        else:
            pass

        return beta, (indices[going_left], indices[going_right])

    def fit(self, X_flattened, y, sample_weight=None):

        X = X_flattened.reshape(X_flattened.shape[0],self.image_size[0],self.image_size[1])
        print(X.shape)


        if sample_weight is None:
            sample_weight = np.ones(len(X))

        self.data = []
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.betas = []
        self.proportions = []

        X_gpu = Variable(torch.from_numpy(X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])).float())
        if torch.cuda.is_available():
            X_gpu = X_gpu.cuda()

        print('creating grid')

        full_grid = np.array([np.random.random((self.kernel_shape, self.kernel_shape)) for i in range(10000)])

        for layer in range(self.depth):
            if layer == 0:
                b, splits = self._find_optimal_beta(X_gpu, np.arange(len(X)), y, sample_weight, full_grid)
                print("splits", len(splits[0]), len(splits[-1]))
                self.betas.append([b])
                self.data.append([splits])

            else:
                for i in range(len(self.betas[layer-1])):
                    left = self.data[layer-1][i][0]
                    right = self.data[layer-1][i][1]

                    left_beta, left_children = self._find_optimal_beta(X_gpu, left, y, sample_weight, full_grid)
                    print("going left", len(left_children[0]), len(left_children[1]))
                    
                    right_beta, right_children = self._find_optimal_beta(X_gpu, right, y, sample_weight, full_grid)
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
            left_output = [np.average(y.take(left) == c, weights=sample_weight.take(left), axis=0) for c in self.classes_]
            right_output = [np.average(y.take(right) == c, weights=sample_weight.take(right), axis=0) for c in self.classes_]
            self.proportions.extend([left_output, right_output])

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

#    def final_leaf(self, X):
#
#        X = X_flattened.reshape(X_flattened.shape[0],28,28)
#
#        X_pytorch = Variable(torch.from_numpy(X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])).float())
#        
#        flattened_betas = np.array(list(itertools.chain(*self.betas)))
#        self.conv_betas.out_channels = len(flattened_betas)
#
#        betas_output = pytorch_conv_exact(X_pytorch, flattened_betas, self.conv_betas, threshold=self.threshold, limit=self.filters_limit)
#
#        output = []
#        offset = 2**(self.depth) - 1
#        for i in range(len(X)):
#            idx = 0
#            for current_layer in range(self.depth):
#                if betas_output[idx, i]:
#                    idx = idx*2 + 1
#                else:
#                    idx = idx*2 + 2
#            #output.append(self.proportions[idx - offset])
#            output.append(idx-offset)
#        
#        return np.array(output)


    def predict_proba(self, X):
        return self.decision_function(X)

    def decision_function(self, X_flattened):
        X = X_flattened.reshape(X_flattened.shape[0],self.image_size[0],self.image_size[1])

        X_pytorch = Variable(torch.from_numpy(X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])).float())
        
        flattened_betas = np.array(list(itertools.chain(*self.betas)))
        self.conv_betas.out_channels = len(flattened_betas)
        betas_output = pytorch_conv_exact2d(X_pytorch, flattened_betas, self.conv_betas, threshold=self.threshold, limit=self.filters_limit)

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
        predicted_probs = self.predict_proba(X)
        return self.classes_.take(np.argmax(predicted_probs,axis=1))


    def score(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        return np.average(self.predict(X) == y, weights=sample_weight, axis=0)






class ConvDTClassifierDNA(BaseEstimator):
    def __init__(self, depth, motif_length, sequence_length, iterations=10, num_processes=4, alpha=0.80, loss_function=two_class_weighted_entropy, optimization_sample_size=(2000,2000), regularization=0):
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
        self.regularization = regularization
                

    def _find_optimal_beta(self, X, X_rc, indices, y, weights, grid, cov_init=0.4, elite_num=20):

        cov = cov_init
        best_memory = None
        best_score = 99999
        best_classifications = None
        
        ### sample members (betas) ###
        if len(indices)==0:
            return np.zeros(4*self.motif_length), ([],[])

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
            #print(indices)
            print('indices shape', indices.shape)
            indices_cuda = Variable(torch.LongTensor(indices))
            if torch.cuda.is_available():
                indices_cuda = indices_cuda.cuda()
            classifications = pytorch_convDNA(X.index_select(dim=0, index=indices_cuda), 
                                           X_rc.index_select(dim=0, index=indices_cuda), members, self.conv, limit=1000)

            member_scores = np.apply_along_axis(self.loss_function,1,better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices]))
            member_scores += self.regularization*np.abs(members).sum(axis=1)


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
        print('RIGHT HERE', classifications.shape)

        if child_variance(y[indices], classifications)[0] > best_score:
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
            left_output = [np.average(y.take(left) == c, 
                weights=sample_weight.take(left),
                axis=0) if len(y.take(left)==c)!=0 else 0 for c in self.classes_]
            right_output = [np.average(y.take(right) == c,
                weights=sample_weight.take(right), 
                axis=0) if len(y.take(right)==c)!=0 else 0 for c in self.classes_]
            self.proportions.extend([left_output, right_output])


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

    def predict_proba(self, X):
        return self.decision_function(X)

    def decision_function(self, X):
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
        predicted_probs = self.predict_proba(X)
        return self.classes_.take(np.argmax(predicted_probs,axis=1))


    def score(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        return np.average(self.predict(X) == y, weights=sample_weight, axis=0)







################
## REGRESSORS ##
################

class ConvDTRegressor2d(BaseEstimator):
    def __init__(self, depth, kernel_shape=5, iterations=12, threshold=3000, num_processes=4, alpha=0.80, loss_function=multi_class_weighted_entropy, optimization_sample_size=(2000,2000), Regression=False):
        self.depth = depth
        self.kernel_shape = kernel_shape
        self.iterations = iterations
        self.alpha = alpha
        self.num_processes = num_processes
        self.loss_function = loss_function
        self.threshold = threshold
        self.data = []
        self.optimization_sample_size = optimization_sample_size
        self.conv = nn.Conv2d(1,100,kernel_size=kernel_shape,bias=False)
        self.conv_single = nn.Conv2d(1,1,kernel_size=kernel_shape,bias=False)
        self.conv_betas = nn.Conv2d(1,1,kernel_size=kernel_shape,bias=False)
        self.Regression=Regression
                

    def _find_optimal_beta(self, X, indices, y, weights, grid, cov_init=0.4, elite_num=20):

        cov = cov_init
        best_memory = None
        best_score = 99999
        best_classifications = None
        
        ### sample members (betas) ###
        for i in range(self.iterations):
            print('iteration:', i)
            if i==0:
                members = grid[np.random.choice(range(len(grid)), size=self.optimization_sample_size[0], replace=False)]
            else:
                members = multivariate_normal.rvs(mean=mu.flatten(), cov=cov, size=self.optimization_sample_size[1]).reshape(self.optimization_sample_size[1],self.kernel_shape, self.kernel_shape)

            print('calculating scores...')
            
            ####################
            ### PYTORCH PART ###
            ####################

            #### TESTING ####
            indices_cuda = Variable(torch.LongTensor(indices))
            if torch.cuda.is_available():
                indices_cuda = indices_cuda.cuda()
            classifications = pytorch_conv(X.index_select(dim=0, index=indices_cuda), members, self.conv, threshold=self.threshold, limit=100)

            # get the entropy scores for each member (beta)
            if self.Regression:
                member_scores = child_variance(np.array(y[indices]), classifications)
            else:
                #print(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices]))
                #print(np.sum(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices]), axis=1))
                member_scores = np.apply_along_axis(self.loss_function,1,better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices]))


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

        self.conv_single.weight.data = torch.from_numpy(mu.reshape(1,1,self.kernel_shape,self.kernel_shape)).float()
        if torch.cuda.is_available():
            self.conv_single = self.conv_single.cuda()
        output_forward = self.conv_single(X.index_select(dim=0, index=indices_cuda))
        #output_rc = self.conv_single(X_rc.index_select(dim=0, index=indices_cuda))
        classifications = np.swapaxes((output_forward.max(dim=2)[0].max(dim=2)[0] >= self.threshold).cpu().data.numpy(),0,1)
        print('RIGHT HERE', classifications.shape)

        #if self.loss_function(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices])[0]) > best_score:
        if child_variance(y[indices], classifications)[0] > best_score:
            print("going with something else")
            beta = best_memory
            output_classifications = best_classifications
        else:
            print("we good")
            beta = mu
            output_classifications = classifications[0]

        return beta, (indices[np.where(output_classifications==1)[0]], indices[np.where(output_classifications==0)[0]])

    def fit(self, X_flattened, y, sample_weight=None):

        X = X_flattened.reshape(X_flattened.shape[0],28,28)
        print(X.shape)


        if sample_weight is None:
            sample_weight = np.ones(len(X))

        self.data = []
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.betas = []
        self.proportions = []

        X_gpu = Variable(torch.from_numpy(X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])).float())
        #Xrc_gpu = Variable(torch.from_numpy(np.array([x[::-1] for x in X]).reshape(X.shape[0], 1, X.shape[1])).float())
        if torch.cuda.is_available():
            X_gpu = X_gpu.cuda()
            #Xrc_gpu = Xrc_gpu.cuda()

        print('creating grid')
        #nucleotides = ['A', 'C', 'G', 'T']
        #keywords = itertools.product(nucleotides, repeat=self.motif_length)
        #kmer_list = ["".join(x) for x in keywords]
        #div = find_best_div(self.sequence_length, self.motif_length, 0.5)
        #full_grid = np.array([motif_to_beta(x) for x in kmer_list]) / div

        full_grid = np.array([np.random.random((self.kernel_shape, self.kernel_shape)) for i in range(10000)])

        for layer in range(self.depth):
            if layer == 0:
                b, splits = self._find_optimal_beta(X_gpu, np.arange(len(X)), y, sample_weight, full_grid)
                print("splits", len(splits[0]), len(splits[-1]))
                self.betas.append([b])
                self.data.append([splits])

            else:
                for i in range(len(self.betas[layer-1])):
                    left = self.data[layer-1][i][0]
                    right = self.data[layer-1][i][1]

                    left_beta, left_children = self._find_optimal_beta(X_gpu, left, y, sample_weight, full_grid)
                    print("going left", len(left_children[0]), len(left_children[1]))
                    
                    right_beta, right_children = self._find_optimal_beta(X_gpu, right, y, sample_weight, full_grid)
                    print("going right", len(right_children[0]), len(right_children[1]))

                    if i==0: #have to append instead of extend on first iteration
                        self.betas.append([left_beta, right_beta])
                        self.data.append([left_children, right_children])
                    else:
                        self.betas[layer].extend([left_beta, right_beta])
                        self.data[layer].extend([left_children, right_children])

        if self.Regression:
            for i in range(len(self.betas[-1])):
                left = self.data[-1][i][0]
                right = self.data[-1][i][1]
                print("LEFT", len(left))
                print("RIGHT", len(right))
                left_output = np.average(y.take(left), weights=sample_weight.take(left), axis=0)
                right_output = np.average(y.take(right), weights=sample_weight.take(right), axis=0)
                self.proportions.extend([left_output, right_output])
        else:
            for i in range(len(self.betas[-1])):
                left = self.data[-1][i][0]
                right = self.data[-1][i][1]
                print("LEFT", len(left))
                print("RIGHT", len(right))
                left_output = np.average(y.take(left) == self.classes_[0], weights=sample_weight.take(left), axis=0)
                right_output = np.average(y.take(right) == self.classes_[0], weights=sample_weight.take(right), axis=0)
                self.proportions.extend([left_output, right_output])

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


    def predict_proba(self, X):
        decision_output = self.decision_function(X)
        if self.Regression:
            proba = expit(decision_output)
            return np.array([(x, 1-x) for x in decision_output])

        else:
            return np.array([(x, 1-x) for x in decision_output])

    def decision_function(self, X_flattened):
        X = X_flattened.reshape(X_flattened.shape[0],28,28)

        X_pytorch = Variable(torch.from_numpy(X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])).float())
        
        flattened_betas = np.array(list(itertools.chain(*self.betas)))
        self.conv_betas.out_channels = len(flattened_betas)
        
        self.conv_betas.weight.data = torch.from_numpy(flattened_betas.reshape(len(flattened_betas),1,self.kernel_shape,self.kernel_shape)).float()
        if torch.cuda.is_available():
            self.conv_betas.cuda()
            X_pytorch = X_pytorch.cuda()
        output1 = self.conv_betas(X_pytorch)
        betas_output = np.swapaxes((output1.max(dim=2)[0].max(dim=2)[0] > self.threshold).cpu().data.numpy(),0,1)
        
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
            
        return np.average(self.predict(X) == y, weights=sample_weight, axis=0)





class ConvDTRegressorDNA(BaseEstimator):
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
        best_score = 99999
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
            classifications = pytorch_convDNA(X.index_select(dim=0, index=indices_cuda), 
                                           X_rc.index_select(dim=0, index=indices_cuda), members, self.conv, limit=1000)

            # get the entropy scores for each member (beta)
            member_scores = child_variance(np.array(y[indices]), classifications)

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
        print('RIGHT HERE', classifications.shape)

        if child_variance(y[indices], classifications)[0] > best_score:
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
            left_output = np.average(y.take(left), weights=sample_weight.take(left), axis=0)
            right_output = np.average(y.take(right), weights=sample_weight.take(right), axis=0)
            self.proportions.extend([left_output, right_output])


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

    #def predict_proba(self, X):
    #    #decision_output = self.decision_function(X)
    #    #if self.Regression:
    #    #    proba = expit(decision_output)
    #    #    return np.array([(x, 1-x) for x in decision_output])

    #    #else:
    #    #    return np.array([(x, 1-x) for x in decision_output])
    #    #return self.decision_function(X)
    #    proba = expit(self.decision_function(X))
    #    return np.array([(x, 1-x) for x in proba])
        


    def decision_function(self, X):
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
        #predicted_probs = self.predict_proba(X)
        #return self.classes_.take(np.argmax(predicted_probs,axis=1))
        return self.decision_function(X)


    def score(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X))
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











