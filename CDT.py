import pandas as pd
import numpy as np
import itertools
from scipy.stats import multivariate_normal, entropy
import scipy
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

from ConvFunctions import *
import Optim as CDTOptim
import Loss as CDTLoss

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


#########################
### CLASS DEFINITIONS ###
#########################

class CDTClassifier(BaseEstimator):
    def __init__(self, depth, filter_size, input_size=None, iterations=10, alpha=0.80, optimization_sample_size=(2000,2000), optimizer=CDTOptim.CEOptimizer(5000,.01,25,32), DNA=False, filter_limit=512):
        self.depth = depth
        self.DNA = DNA
        self.filter_size = filter_size
        self.filter_limit = filter_limit
        self.input_size = input_size
        self.iterations = iterations
        self.alpha = alpha
        self.loss_history = []
        self.data = []
        self.optimization_sample_size = optimization_sample_size

        if self.DNA:
            self.conv_betas = nn.Conv1d(1,1,kernel_size=filter_size*4,stride=4,bias=False)
        else:
            self.conv_betas = nn.Conv2d(1,1,kernel_size=filter_size,bias=False)

        self.optimizer=optimizer

    def fit(self, X, y, sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones(len(X))

        self.data = []
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.betas = []
        self.proportions = []

        if self.DNA:
            X_gpu = torch.from_numpy(np.expand_dims(X, axis=1)).float()
            Xrc_gpu = torch.from_numpy(np.array([x[::-1] for x in X]).reshape(X.shape[0], 1, X.shape[1])).float()
            if torch.cuda.is_available():
                X_gpu = X_gpu.cuda()
                Xrc_gpu = Xrc_gpu.cuda()
        else:
            X_gpu = torch.from_numpy(np.expand_dims(X,axis=1)).float()
            if torch.cuda.is_available():
                X_gpu = X_gpu.cuda()
            Xrc_gpu = None

        for layer in range(self.depth):
            if layer == 0:
                b, splits = self.optimizer.find_optimal_beta(X_gpu, Xrc_gpu, np.arange(len(X)), y, sample_weight)
                print("splits", len(splits[0]), len(splits[-1]))
                self.betas.append([b])
                self.data.append([splits])

            else:
                for i in range(len(self.betas[layer-1])):
                    left = self.data[layer-1][i][0]
                    right = self.data[layer-1][i][1]

                    left_beta, left_children = self.optimizer.find_optimal_beta(X_gpu, Xrc_gpu, left, y, sample_weight)
                    print("going left", len(left_children[0]), len(left_children[1]))
                    
                    right_beta, right_children = self.optimizer.find_optimal_beta(X_gpu, Xrc_gpu, right, y, sample_weight)
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
            out = classify_sequence(x, self.betas[current_layer][position], self.filter_size)
            if out==1:#self.classes_[0]: ## if motif found, go to left node
                position = position*2
            else:
                position = position*2+1
                
        return self.proportions[position]

    def predict_proba(self, X):
        return self.decision_function(X)

    def decision_function(self, X):
        X_pytorch = torch.from_numpy(np.expand_dims(X,axis=1)).float()
        if self.DNA:
            Xrc_pytorch = torch.from_numpy(np.expand_dims(np.array([x[::-1] for x in X]),axis=1)).float()
        else:
            Xrc_pytorch = None
        
        flattened_betas = np.array(list(itertools.chain(*self.betas)))
        self.conv_betas.out_channels = len(flattened_betas)
        
        if self.DNA:
            self.conv_betas.weight.data = torch.from_numpy(flattened_betas.reshape(len(flattened_betas),1,len(flattened_betas[0]))).float()
            if torch.cuda.is_available():
                self.conv_betas.cuda()
                X_pytorch = X_pytorch.cuda()
                Xrc_pytorch = Xrc_pytorch.cuda()
            output1 = self.conv_betas(X_pytorch)
            output2 = self.conv_betas(Xrc_pytorch)
            betas_output = np.swapaxes((torch.max(output1, output2).max(dim=2)[0] >= self.optimizer.threshold).cpu().data.numpy(),0,1)
        else:
            betas_output = pytorch_conv_exact2d(X_pytorch, flattened_betas, self.conv_betas, threshold=self.optimizer.threshold, limit=self.filter_limit)
        
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

class CDTRegressor(BaseEstimator):
    def __init__(self, depth, filter_size, input_size=None, iterations=10, alpha=0.80, optimization_sample_size=(2000,2000), optimizer=CDTOptim.CEOptimizer(5000,.01,25,32), DNA=False, filter_limit=1000):
        self.depth = depth
        self.DNA = DNA
        self.filter_size = filter_size
        self.filter_limit = filter_limit
        self.input_size = input_size
        self.iterations = iterations
        self.alpha = alpha
        self.loss_history = []
        self.data = []
        self.optimization_sample_size = optimization_sample_size

        if self.DNA:
            self.conv_betas = nn.Conv1d(1,1,kernel_size=filter_size*4,stride=4,bias=False)
        else:
            self.conv_betas = nn.Conv2d(1,1,kernel_size=filter_size,bias=False)

        self.optimizer=optimizer

    def fit(self, X, y, sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones(len(X))

        self.data = []
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.betas = []
        self.proportions = []

        if self.DNA:
            X_gpu = torch.from_numpy(np.expand_dims(X, axis=1)).float()
            Xrc_gpu = torch.from_numpy(np.array([x[::-1] for x in X]).reshape(X.shape[0], 1, X.shape[1])).float()
            if torch.cuda.is_available():
                X_gpu = X_gpu.cuda()
                Xrc_gpu = Xrc_gpu.cuda()
        else:
            X_gpu = torch.from_numpy(np.expand_dims(X,axis=1)).float()
            if torch.cuda.is_available():
                X_gpu = X_gpu.cuda()
            Xrc_gpu = None

        for layer in range(self.depth):
            if layer == 0:
                b, splits = self.optimizer.find_optimal_beta(X_gpu, Xrc_gpu, np.arange(len(X)), y, sample_weight)
                print("splits", len(splits[0]), len(splits[-1]))
                self.betas.append([b])
                self.data.append([splits])

            else:
                for i in range(len(self.betas[layer-1])):
                    left = self.data[layer-1][i][0]
                    right = self.data[layer-1][i][1]

                    left_beta, left_children = self.optimizer.find_optimal_beta(X_gpu, Xrc_gpu, left, y, sample_weight)
                    print("going left", len(left_children[0]), len(left_children[1]))
                    
                    right_beta, right_children = self.optimizer.find_optimal_beta(X_gpu, Xrc_gpu, right, y, sample_weight)
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
            out = classify_sequence(x, self.betas[current_layer][position], self.filter_size)
            if out==1:#self.classes_[0]: ## if motif found, go to left node
                position = position*2
            else:
                position = position*2+1
                
        return self.proportions[position]

    def predict_proba(self, X):
        return self.decision_function(X)

    def decision_function(self, X):
        X_pytorch = torch.from_numpy(np.expand_dims(X,axis=1)).float()
        if self.DNA:
            Xrc_pytorch = torch.from_numpy(np.expand_dims(np.array([x[::-1] for x in X]),axis=1)).float()
        else:
            Xrc_pytorch = None
        
        flattened_betas = np.array(list(itertools.chain(*self.betas)))
        self.conv_betas.out_channels = len(flattened_betas)
        
        if self.DNA:
            self.conv_betas.weight.data = torch.from_numpy(flattened_betas.reshape(len(flattened_betas),1,len(flattened_betas[0]))).float()
            if torch.cuda.is_available():
                self.conv_betas.cuda()
                X_pytorch = X_pytorch.cuda()
                Xrc_pytorch = Xrc_pytorch.cuda()
            output1 = self.conv_betas(X_pytorch)
            output2 = self.conv_betas(Xrc_pytorch)
            betas_output = np.swapaxes((torch.max(output1, output2).max(dim=2)[0] >= self.optimizer.threshold).cpu().data.numpy(),0,1)
        else:
            betas_output = pytorch_conv_exact2d(X_pytorch, flattened_betas, self.conv_betas, threshold=self.optimizer.threshold, limit=self.filter_limit)
        
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
        return self.predict_proba(X)


