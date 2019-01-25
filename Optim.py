import numpy as np
from ConvFunctions import *
import pandas as pd
from scipy.stats import multivariate_normal, entropy
import scipy
from scipy.special import expit
import torch.nn as nn
import itertools


class GradientDescentOptimizer():

    def __init__(self,param_dimensions,loss_function,iterations=0,step_size=1,alpha=1,classes=np.array([0,1])):
        self.iterations = iterations
        self.step_size = step_size
        self.alpha = alpha
        self.size = param_dimensions
        self.conv_single = nn.Conv1d(1,1,kernel_size=param_dimensions,stride=4,bias=False)
        self.loss_function = loss_function
        self.classes_ = classes
        self.loss_history = []

    def _initialize_beta(self):

        nucleotides = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
        best_div = find_best_div(200, self.size//4, 0.5)
        initial_beta = (np.array([nucleotides[np.random.randint(4)] for _ in range(self.size//4)])/best_div).flatten()

        print(initial_beta)
        return initial_beta

    def find_optimal_beta(self, X, X_rc, indices, y, weights):

        beta = self._initialize_beta()

        if len(indices)==0:
            return np.zeros(self.size), ([], [])

        for i in range(self.iterations):
            #print(f'beta: {beta}')
            indices_cuda = Variable(torch.LongTensor(indices))
            if torch.cuda.is_available():
                indices_cuda = indices_cuda.cuda()
                
            self.conv_single.weight.data = torch.from_numpy(beta.reshape(1,1,self.size)).float()
            if torch.cuda.is_available():
                self.conv_single = self.conv_single.cuda()
            output_forward = self.conv_single(X.index_select(dim=0, index=indices_cuda))
            output_rc = self.conv_single(X_rc.index_select(dim=0, index=indices_cuda))
            classifications = np.swapaxes((torch.max(output_forward, output_rc).max(dim=2)[0] >= 1).cpu().data.numpy(),0,1)

            #if i%10==0:
            #    print(f'iteration: {i}')
            #    print(self.loss_function(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices])[0]))

            max_vals, max_sequences = convDNA_single_maxinfo(X.index_select(dim=0, index=indices_cuda), X_rc.index_select(dim=0, index=indices_cuda), beta)
            max_vals_expit = expit(self.alpha*(max_vals-1))

            #print('LEFT AND RIGHT: ', len(np.where(max_vals > 1)[0]), len(np.where(max_vals <= 1)[0]))

            P = ((y[indices]==1)).sum()
            N = ((y[indices]==0)).sum()

            p = (max_vals_expit * (y[indices]==1)).sum()
            n = (max_vals_expit * (y[indices]==0)).sum()

            p_prime = self.alpha*(max_sequences.T * (((max_vals_expit)*(1-max_vals_expit)) * (y[indices].values==1))).T.sum(axis=0)
            n_prime = self.alpha*(max_sequences.T * (((max_vals_expit)*(1-max_vals_expit)) * (y[indices].values==0))).T.sum(axis=0)

            if i%30==0:
                print(f'iteration: {i}')
                loss = self.loss_function(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices])[0])
                self.loss_history.append(loss)
                print('loss function: ', loss)
                print(f'Information Gain {IG(P,N,p,n)}')
                #print(beta)

            gradient = (1/(P+N))*(-(p_prime + n_prime)*np.log((p+n)/(P+N-p-n)) + p_prime*np.log(p/(P-p)) + n_prime*np.log(n/(N-n)))
            beta += self.step_size*gradient

        max_vals, max_sequences = convDNA_single_maxinfo(X.index_select(dim=0, index=indices_cuda), X_rc.index_select(dim=0, index=indices_cuda), beta.reshape(1,-1))

        return beta, (np.where(max_vals > 1)[0], np.where(max_vals <= 1)[0])




class CEOptimizer():
    
    def __init__(self, loss_function, motif_length, sequence_length, iterations=10, optimization_sample_size=(1000,1000), elite_num=20, cov_init=0.4, classes=np.array([0,1]), alpha=0.8):
        self.iterations = iterations
        self.cov_init = cov_init
        self.optimization_sample_size = optimization_sample_size
        self.elite_num = elite_num
        self.loss_function = loss_function
        self.conv_single = nn.Conv1d(1,1,kernel_size=motif_length*4,stride=4,bias=False)
        self.conv = nn.Conv1d(1,1000,kernel_size=motif_length*4,stride=4,bias=False)
        self.motif_length = motif_length
        self.sequence_length = sequence_length
        self.classes_ = classes
        self.alpha = alpha
        self.loss_history = []

        print('creating grid')
        nucleotides = ['A', 'C', 'G', 'T']
        keywords = itertools.product(nucleotides, repeat=self.motif_length)
        kmer_list = ["".join(x) for x in keywords]
        div = find_best_div(self.sequence_length, self.motif_length, 0.5)
        self.grid = np.array([motif_to_beta(x) for x in kmer_list]) / div

    def find_optimal_beta(self, X, X_rc, indices, y, weights):

        cov = self.cov_init
        best_memory = None
        best_score = np.inf
        best_classifications = None

        ### sample members (betas) ###
        if len(indices)==0:
            return np.zeros(self.motif_length*4), ([],[])

        for i in range(self.iterations):
            print('iteration:', i)
            if i==0:
                members = self.grid[np.random.choice(range(len(self.grid)), size=self.optimization_sample_size[0], replace=False)]
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
            #member_scores += self.regularization*np.abs(members).sum(axis=1)


            #print('getting best scores')
            best_scoring_indices = np.argsort(member_scores)[0:self.elite_num]
            if member_scores[best_scoring_indices[0]] < best_score:
                best_score = member_scores[best_scoring_indices[0]]
                best_memory = members[best_scoring_indices[0]]
                best_classifications = classifications[best_scoring_indices[0]]
            else:
                pass
            self.loss_history.append(best_score)
            
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

        #if child_variance(y[indices], classifications)[0] > best_score:
        #print(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices]))
        #print(self.loss_function(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices])))
        if self.loss_function(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices])[0]) > best_score:
            print("going with something else")
            beta = best_memory
            output_classifications = best_classifications
        else:
            print("we good")
            beta = mu
            output_classifications = classifications[0]

        return beta, (indices[np.where(output_classifications==1)[0]], indices[np.where(output_classifications==0)[0]])


class SimulatedAnnealingOptimizer():

    def __init__(self, loss_function, iterations, motif_length, sequence_length, T_initial, cooling_factor, step_size, classes=np.array([0,1])):
        self.iterations = iterations
        self.loss_function = loss_function
        self.conv_single = nn.Conv1d(1,1,kernel_size=motif_length*4,stride=4,bias=False)
        self.motif_length = motif_length
        self.T_initial = T_initial
        self.step_size = step_size
        self.cooling_factor = cooling_factor
        self.sequence_length = sequence_length
        self.classes_ = classes
        self.loss_history = []
    
    def _initialize_beta(self):
        nucleotides = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
        best_div = find_best_div(200, self.motif_length, 0.5)
        initial_beta = (np.array([nucleotides[np.random.randint(4)] for _ in range(self.motif_length)])/best_div).flatten()

        print(initial_beta)
        return initial_beta

    def find_optimal_beta(self, X, X_rc, indices, y, weights):
        beta = self._initialize_beta()

        indices_cuda = Variable(torch.LongTensor(indices))
        if torch.cuda.is_available():
            indices_cuda = indices_cuda.cuda()
            
        self.conv_single.weight.data = torch.from_numpy(beta.reshape(1,1,self.motif_length*4)).float()
        if torch.cuda.is_available():
            self.conv_single = self.conv_single.cuda()
        output_forward = self.conv_single(X.index_select(dim=0, index=indices_cuda))
        output_rc = self.conv_single(X_rc.index_select(dim=0, index=indices_cuda))
        classifications = np.swapaxes((torch.max(output_forward, output_rc).max(dim=2)[0] >= 1).cpu().data.numpy(),0,1)

        current_cost = self.loss_function(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices])[0])

        self.loss_history.append(current_cost)

        T = self.T_initial

        for i in range(self.iterations):
            print(f'current cost, {current_cost}')
            update_beta = np.random.normal(beta, self.step_size) 

            self.conv_single.weight.data = torch.from_numpy(beta.reshape(1,1,self.motif_length*4)).float()
            if torch.cuda.is_available():
                self.conv_single = self.conv_single.cuda()
            output_forward = self.conv_single(X.index_select(dim=0, index=indices_cuda))
            output_rc = self.conv_single(X_rc.index_select(dim=0, index=indices_cuda))
            classifications = np.swapaxes((torch.max(output_forward, output_rc).max(dim=2)[0] >= 1).cpu().data.numpy(),0,1)

            update_cost = self.loss_function(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices])[0])

            if update_cost < current_cost:
                print("updated")
                beta = update_beta
                current_cost = update_cost

            else:
                transition_probability = np.exp((current_cost - update_cost)/T)
                print(f'current cost {current_cost} and update cost {update_cost}')
                print(transition_probability)
                if np.random.random() < transition_probability:
                    print("updated")
                    beta = update_beta
                    current_cost = update_cost

            self.loss_history.append(current_cost)

            T *= self.cooling_factor

        self.conv_single.weight.data = torch.from_numpy(beta.reshape(1,1,self.motif_length*4)).float()
        if torch.cuda.is_available():
            self.conv_single = self.conv_single.cuda()
        output_forward = self.conv_single(X.index_select(dim=0, index=indices_cuda))
        output_rc = self.conv_single(X_rc.index_select(dim=0, index=indices_cuda))
        classifications = np.swapaxes((torch.max(output_forward, output_rc).max(dim=2)[0] >= 1).cpu().data.numpy(),0,1)

        return beta, (indices[np.where(classifications==1)[0]], indices[np.where(classifications==0)[0]])


