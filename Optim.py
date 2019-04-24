import numpy as np
from ConvFunctions import *
import pandas as pd
from scipy.stats import multivariate_normal, entropy
import scipy
from scipy.special import expit
import torch.nn as nn
import itertools



class GradientDescentOptimizer():

    def __init__(self,motif_length,sequence_length,loss_function,iterations=0,step_size=1,alpha=1,classes=np.array([0,1]),init_sequence=None):
        self.iterations = iterations
        self.step_size = step_size
        self.alpha = alpha
        self.motif_length = motif_length
        self.sequence_length = sequence_length
        self.conv_single = nn.Conv1d(1,1,kernel_size=motif_length*4,stride=4,bias=False)
        self.loss_function = loss_function
        self.classes_ = classes
        self.loss_history = []
        self.beta_history = []
        self.init_sequence = init_sequence

    def _initialize_beta(self):
        best_div = find_best_div(self.sequence_length, self.motif_length, 0.5)

        if self.init_sequence:
            initial_beta = motif_to_beta(self.init_sequence) / best_div
        else:
            nucleotides = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
            initial_beta = (np.array([nucleotides[np.random.randint(4)] for _ in range(self.motif_length)])/best_div).flatten()

        print(initial_beta)
        return initial_beta

    def find_optimal_beta(self, X, X_rc, indices, y, weights):

        beta_history = []
        loss_history = []
        beta = self._initialize_beta()

        if len(indices)==0:
            return np.zeros(self.motif_length*4), ([], [])

        for i in range(self.iterations):
            #print(f'beta: {beta}')
            indices_cuda = Variable(torch.LongTensor(indices))
            if torch.cuda.is_available():
                indices_cuda = indices_cuda.cuda()
                
            self.conv_single.weight.data = torch.from_numpy(beta.reshape(1,1,-1)).float()
            if torch.cuda.is_available():
                self.conv_single = self.conv_single.cuda()
            output_forward = self.conv_single(X.index_select(dim=0, index=indices_cuda))
            output_rc = self.conv_single(X_rc.index_select(dim=0, index=indices_cuda))
            classifications = np.swapaxes((torch.max(output_forward, output_rc).max(dim=2)[0] >= 1).cpu().data.numpy(),0,1)

            max_vals, max_sequences = convDNA_single_maxinfo(X.index_select(dim=0, index=indices_cuda), X_rc.index_select(dim=0, index=indices_cuda), beta)
            max_vals_expit = expit(self.alpha*(max_vals-1))

            #print('LEFT AND RIGHT: ', len(np.where(max_vals > 1)[0]), len(np.where(max_vals <= 1)[0]))

            P = ((y[indices]==1) * weights[indices]).sum()
            N = ((y[indices]==0) * weights[indices]).sum()

            p = (max_vals_expit * (y[indices]==1) * weights[indices]).sum()
            n = (max_vals_expit * (y[indices]==0) * weights[indices]).sum()

            p_true = ((classifications==1)[0] * (y[indices]==1) * weights[indices]).sum()
            n_true = ((classifications==1)[0] * (y[indices]==0) * weights[indices]).sum()

            p_prime = self.alpha*(max_sequences.T * (((max_vals_expit)*(1-max_vals_expit)) * (y[indices].values==1) * weights[indices])).T.sum(axis=0)
            n_prime = self.alpha*(max_sequences.T * (((max_vals_expit)*(1-max_vals_expit)) * (y[indices].values==0) * weights[indices])).T.sum(axis=0)

            if i%100==0:
                #print(f'iteration: {i}')
                loss = self.loss_function(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices])[0])
                loss_history.append(loss)
                #loss_history.append(IG(P,N,p,n))
                beta_history.append(beta)
                #print('loss function: ', loss)
                #print(f'Information Gain {IG(P,N,p,n)}')
                #print(f'Real Information Gain {IG(P,N,p_true,n_true)}')


            gradient = (1/(P+N))*(-(p_prime + n_prime)*np.log((p+n)/(P+N-p-n)) + p_prime*np.log(p/(P-p)) + n_prime*np.log(n/(N-n)))
            beta = beta + self.step_size*gradient
        
        beta_history.append(beta)
        self.beta_history.append(beta_history)
        self.loss_history.append(loss_history)

        max_vals, max_sequences = convDNA_single_maxinfo(X.index_select(dim=0, index=indices_cuda), X_rc.index_select(dim=0, index=indices_cuda), beta.reshape(1,-1))

        #print((np.where(max_vals > 1)[0], np.where(max_vals <= 1)[0]))
        return beta, (indices[np.where(max_vals > 1)[0]], indices[np.where(max_vals <= 1)[0]])



class CEOptimizer():
    
    def __init__(self, loss_function, filter_size, input_size=None, iterations=10, optimization_sample_size=(1000,1000), elite_num=20, cov_init=0.4, classes=np.array([0,1]), alpha=0.8, smart_init=True, DNA=False, threshold=1, filters_limit=512):
        self.iterations = iterations
        self.cov_init = cov_init
        self.optimization_sample_size = optimization_sample_size
        self.elite_num = elite_num
        self.loss_function = loss_function
        self.filter_size = filter_size
        self.input_size = input_size
        self.classes_ = classes
        self.alpha = alpha
        self.loss_history = []
        self.beta_history = []
        self.smart_init = smart_init
        self.DNA = DNA
        self.threshold = threshold
        self.filters_limit = filters_limit

        if self.DNA:
            self.conv_single = nn.Conv1d(1,1,kernel_size=filter_size*4,stride=4,bias=False)
            self.conv = nn.Conv1d(1,1000,kernel_size=filter_size*4,stride=4,bias=False)
            
            print('creating grid')
            nucleotides = ['A', 'C', 'G', 'T']
            keywords = itertools.product(nucleotides, repeat=self.filter_size)
            kmer_list = ["".join(x) for x in keywords]
            div = find_best_div(self.input_size, self.filter_size, 0.5) / self.threshold
            self.grid = np.array([motif_to_beta(x) for x in kmer_list]) / div
        else:
            self.conv = nn.Conv2d(1,filters_limit,kernel_size=filter_size,bias=False)
            self.conv_single = nn.Conv2d(1,1,kernel_size=filter_size,bias=False)

    
    def _initialize_CE(self):
        if self.DNA:
            members = self.grid[np.random.choice(range(len(self.grid)), size=self.optimization_sample_size[0], replace=False)]
        else:
            members = np.array([np.random.random(self.filter_size) for i in range(self.optimization_sample_size[0])])

        return members


    def find_optimal_beta(self, X, X_rc, indices, y, weights):

        cov = self.cov_init
        best_memory = None
        best_score = np.inf
        best_classifications = None

        beta_history = []
        loss_history = []

        ### sample members (betas) ###
        if len(indices)==0:
            if self.DNA:
                return np.zeros(self.filter_size*4), ([],[])
            else:
                return np.zeros(self.filter_size), ([],[])

        for i in range(self.iterations):
            print('iteration:', i)
            if i==0:
                if self.smart_init:
                    #members = self.grid[np.random.choice(range(len(self.grid)), size=self.optimization_sample_size[0], replace=False)]
                    members = self._initialize_CE()
                else:
                    mu = self.grid[np.random.randint(len(self.grid))]
                    beta_history.append(mu)
                    members = multivariate_normal.rvs(mean=mu, cov=cov, size=self.optimization_sample_size[0])
            else:
                if type(self.filter_size)==int:
                    members = multivariate_normal.rvs(mean=mu.flatten(), cov=cov, size=self.optimization_sample_size[1])
                else:
                    members = multivariate_normal.rvs(mean=mu.flatten(), cov=cov, size=self.optimization_sample_size[1]).reshape((-1,)+self.filter_size)

            print('calculating scores...')
            
            ####################
            ### PYTORCH PART ###
            ####################

            #print(indices)
            print('indices shape', indices.shape)
            indices_cuda = torch.LongTensor(indices)
            if torch.cuda.is_available():
                indices_cuda = indices_cuda.cuda()

            if X_rc is not None:
                classifications = pytorch_convDNA(X.index_select(dim=0, index=indices_cuda), 
                                           X_rc.index_select(dim=0, index=indices_cuda), members, self.conv, threshold=self.threshold, limit=self.filters_limit)
            else:
                classifications = pytorch_conv2d(X.index_select(dim=0, index=indices_cuda),
                                            members, self.conv, threshold=self.threshold, limit=self.filters_limit)


            #member_scores = np.apply_along_axis(self.loss_function,1,better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices]))
            member_scores = np.apply_along_axis(self.loss_function,1,classifications,y[indices],weights[indices])

            #print('getting best scores')
            best_scoring_indices = np.argsort(member_scores)[0:self.elite_num]
            if member_scores[best_scoring_indices[0]] < best_score:
                best_score = member_scores[best_scoring_indices[0]]
                best_memory = members[best_scoring_indices[0]]
                best_classifications = classifications[best_scoring_indices[0]]
            else:
                pass
            loss_history.append(best_score)

            
            print('best score so far:', best_score)

            ## Calculate the MLE ##
            new_mu = np.mean(members[best_scoring_indices], axis=0)
            new_cov = np.mean([np.outer(x,x) for x in (members[best_scoring_indices] - new_mu)], axis=0) ## maybe faster way

            if i==0:
                mu = new_mu
                print(mu)
                print(mu.shape)
                cov = self.alpha*new_cov + (1-self.alpha)*cov
            else:
                mu = self.alpha*new_mu + (1-self.alpha)*mu
                cov = self.alpha*new_cov + (1-self.alpha)*cov
            
            beta_history.append(mu)

        if X_rc is not None:
            classifications = pytorch_convDNA_single(X.index_select(dim=0, index=indices_cuda), 
                X_rc.index_select(dim=0, index=indices_cuda),
                mu.reshape(1,1,self.filter_size*4),
                self.conv_single, threshold= self.threshold)
        else:
            classifications = pytorch_conv2d_single(X.index_select(dim=0, index=indices_cuda),
                                                mu.reshape((1,1)+self.filter_size),
                                                self.conv_single,
                                                threshold=self.threshold)

        print('RIGHT HERE', classifications.shape)

        if self.loss_function(classifications[0], y[indices], weights[indices]) > best_score:
            print("going with something else")
            beta = best_memory
            output_classifications = best_classifications
        else:
            print("we good")
            beta = mu
            output_classifications = classifications[0]

        self.beta_history.append(beta_history)
        self.loss_history.append(loss_history)

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
        self.beta_history = []
    
    def _initialize_beta(self):
        nucleotides = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
        best_div = find_best_div(200, self.motif_length, 0.5)
        initial_beta = (np.array([nucleotides[np.random.randint(4)] for _ in range(self.motif_length)])/best_div).flatten()

        print(initial_beta)
        return initial_beta

    def _propose_new_smallest(self, beta):
        index = np.random.randint(len(beta))
        proposed_beta = beta.copy()
        proposed_beta[index] += np.random.uniform(low=-self.step_size, high=self.step_size)

        return proposed_beta

    def _propose_new_small(self, beta):
        proposed_beta = beta.copy()
        index = np.random.randint(len(beta)//4)
        proposed_beta[index*4:(index+1)*4] += np.random.uniform(low=-self.step_size, high=self.step_size, size=4)

        return proposed_beta

    def find_optimal_beta(self, X, X_rc, indices, y, weights):
        beta = self._initialize_beta()
        beta_history = []
        loss_history = []

        if len(indices)==0:
            return np.zeros(self.motif_length*4), ([],[])

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

        loss_history.append(current_cost)

        T = self.T_initial

        for i in range(self.iterations):
            if i%100==0:
                beta_history.append(beta)
                #print(f'current cost, {current_cost}')
            #update_beta = np.random.normal(beta, self.step_size) 
            update_beta = self._propose_new_smallest(beta)
            #update_beta = self._propose_new_small(beta)

            self.conv_single.weight.data = torch.from_numpy(update_beta.reshape(1,1,self.motif_length*4)).float()
            if torch.cuda.is_available():
                self.conv_single = self.conv_single.cuda()
            output_forward = self.conv_single(X.index_select(dim=0, index=indices_cuda))
            output_rc = self.conv_single(X_rc.index_select(dim=0, index=indices_cuda))
            classifications = np.swapaxes((torch.max(output_forward, output_rc).max(dim=2)[0] >= 1).cpu().data.numpy(),0,1)

            update_cost = self.loss_function(better_return_counts_weighted(y[indices], classifications, self.classes_, weights[indices])[0])

            if update_cost < current_cost:
                #print("updated")
                beta = update_beta
                current_cost = update_cost

            else:
                transition_probability = np.exp((current_cost - update_cost)/T)
                if np.random.random() < transition_probability:
                    beta = update_beta
                    current_cost = update_cost

            loss_history.append(current_cost)
            T *= self.cooling_factor

        self.conv_single.weight.data = torch.from_numpy(beta.reshape(1,1,self.motif_length*4)).float()
        if torch.cuda.is_available():
            self.conv_single = self.conv_single.cuda()
        output_forward = self.conv_single(X.index_select(dim=0, index=indices_cuda))
        output_rc = self.conv_single(X_rc.index_select(dim=0, index=indices_cuda))
        classifications = np.swapaxes((torch.max(output_forward, output_rc).max(dim=2)[0] >= 1).cpu().data.numpy(),0,1)

        self.beta_history.append(beta_history) 
        self.loss_history.append(loss_history)

        return beta, (indices[np.where(classifications[0]==1)[0]], indices[np.where(classifications[0]==0)[0]])


