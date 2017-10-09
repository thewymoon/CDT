import pandas as pd
import numpy as np
import copy 
from numba import jit, vectorize
import itertools
from numpy.lib.stride_tricks import as_strided
from scipy.stats import multivariate_normal
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

#import nltk


### FUNCTION DEFINITIONS ####

# Loss Function
def entropy(p_vec, pseudo=0.00001):
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
    #return np.logical_or(np.any(np.dot(X_matrices, beta) > 1, axis=1), np.any(np.dot(X_matrices_rc, beta) > 1, axis=1)).astype(int)
    #return np.logical_or(np.any(np.array([np.dot(x, beta) for x in X_matrices]) > 1, axis=1), np.any(np.array([np.dot(x, beta) for x in X_matrices_rc]) > 1, axis=1)).astype(int)
    return np.logical_or(np.any(faster_dot(X_matrices, beta) > 1.0, axis=1), np.any(faster_dot(X_matrices_rc, beta) > 1.0, axis=1)).astype(int)

def weighted_classify_sequences(X_matrices, X_matrices_rc, beta, weights):
    #return np.logical_or(np.any(np.dot(X_matrices, beta) > 1, axis=1), np.any(np.dot(X_matrices_rc, beta) > 1, axis=1)).astype(int)
    #return np.logical_or(np.any(np.array([np.dot(x, beta) for x in X_matrices]) > 1, axis=1), np.any(np.array([np.dot(x, beta) for x in X_matrices_rc]) > 1, axis=1)).astype(int)
    return np.logical_or(np.any(faster_dot(X_matrices, beta) > 1.0, axis=1), np.any(faster_dot(X_matrices_rc, beta) > 1.0, axis=1)).astype(int)
    
def classify_sequence(x, beta, motif_length, sequence_length):
    x_matrix = x_to_matrix(x, motif_length, sequence_length)
    return int(np.any(np.dot(x_matrix, beta) > 1.0) or np.any(np.dot(x_matrix, flip_beta(beta)) > 1.0))

    

@vectorize('float64(float64)')
def threshold(value):
    if value > 0.5:
        return 1
    else:
        return 0



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

def return_weightedcounts(labels, classifications, weights):
    zipped = list(zip(list(zip(labels, classifications)), weights))
    true1 = np.sum([a[1] for a in zipped if a[0]==(1,1)])
    false1 = np.sum([a[1] for a in zipped if a[0]==(0,1)]) 
    true0 = np.sum([a[1] for a in zipped if a[0]==(0,0)])
    false0 = np.sum([a[1] for a in zipped if a[0]==(1,0)])
    return [true1, false1, true0, false0]




def x_to_string(x):
    return "".join([str(i) for i in x])


def x_to_matrix(x, motif_length, sequence_length):
    numpy_arrayx = np.array(x)
    size = numpy_arrayx.itemsize

    #print('size', size)
    return as_strided(numpy_arrayx, shape = [sequence_length - motif_length, motif_length*4], strides = [size*4,size])

#### N GRAM COUNTIN #####
#def getSequenceNgrams(sequence):
#    return [''.join(x) for x in nltk.ngrams(sequence, 6)]
#
#def getCounts(sequences):
#    count_dict = {}
#    ngrams = sequences.apply(getSequenceNgrams)
#    for i in sequences.index.values:
#        for gram in ngrams.ix[i]:
#            if gram not in count_dict.keys():
#                count_dict[gram] = 1
#            else:
#                count_dict[gram] += 1
#    return count_dict

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
            #return two_class_weighted_entropy(return_counts(y, classify_sequences(X_matrices, X_matrices_rc, m)))
            #return two_class_weighted_entropy(return_counts_general(y, classify_sequences(X_matrices, X_matrices_rc, m), classes))
            return two_class_weighted_entropy(return_counts_weighted(y, classify_sequences(X_matrices, X_matrices_rc, m), classes, weights))

#########################
### CLASS DEFINITIONS ###
#########################

class Node:

    def __init__(self, motif_length, seq_length, beta0):
        self.motif_length = motif_length
        self.seq_length = seq_length
        self.thresh = 1
        self.beta = beta0
        self.loss_func = two_class_weighted_entropy
        self.terminal = False
        self.left_classification = None
        self.right_classification = None
        self.loss_memory = []
        self.beta_memory = []
        self.left_child_output = 0.5
        self.right_child_output = 0.5

    def set_loss_function(loss):
        self.loss_func = loss

    def set_terminal_status(self, status):
        self.terminal = status


    def gradientfit(self, X_matrices, X_matrices_RC, y, weights, iterations, step_size):
        data_size = len(X_matrices)
        labels = y


        #X_matrices = [x_to_matrix(x, self.motif_length, self.seq_length) for x in np.array(X)]

        for i in range(iterations):
            grad = weightedgradient(X_matrices,  y, weights, self.beta, self.motif_length, self.seq_length, step_size)

            self.beta += grad[0]
            self.loss_memory.append(self.loss_func(grad[1]))


        classification = classify_sequences(X_matrices, X_matrices_RC, self.beta)
        print("counts...", return_counts(labels, classification))
        current_entropy = self.loss_func(return_counts(labels, classification))
        print("current entropy...", current_entropy)


        ## ONCE FIT, final classification of resulting nodes are defined ##
        final_counts = return_counts(labels, classification)
        if final_counts[0] > final_counts[1]:
            self.left_classification = 1
        else:
            self.left_classification = 0

        if final_counts[2] > final_counts[3]:
            self.right_classification = 0
        else:
            self.right_classification = 1



    def anneal(self, X_matrices, y, weights, alpha=0.9, T_start = .001, T_min = 0.0005, iterT=100):
        #X_matrices = [x_to_matrix(x, self.motif_length, self.seq_length) for x in np.array(X)]
        #cost = self.loss_func(return_weightedcounts(y, classify_sequences(X_matrices, self.beta, self.motif_length, self.seq_length), weights))
        cost = self.loss_func(return_weightedcounts(y, classify_sequences(X_matrices, X_matrices_RC, self.beta), weights))
        T = T_start

        while T > T_min:
            i = 1
            print('New Temperature', T, "\n")
            while i <= iterT:
                ## VERY IMPORTANT STEP!!! POTENTIAL GAINS HERE in better proposals##

                ### Try new beta ###
                while True:
                    new_beta = small_change(self.beta, std=np.random.chisquare(.5))
                    #new_beta = random_change(self.beta, std=np.random.chisquare(.4))
                    if acceptable_beta(new_beta,thresh=1):
                        break

                #new_cost = self.loss_func(return_weightedcounts(y, classify_sequences(X_matrices, new_beta, self.motif_length, self.seq_length), weights))
                new_cost = self.loss_func(return_weightedcounts(y, classify_sequences(X_matrices, X_matrices_RC, new_beta), weights))
                ap = acceptance_probability(cost, new_cost, T)
                #print(ap)
                if ap > np.random.random():
                    self.beta = new_beta
                    cost = new_cost
                    self.loss_memory.append(cost)
                i += 1
            T *= alpha

    
    def crossentropyFit(self, X_matrices, X_matrices_rc, y, weights, alpha, lam, cov_init, iterations, num_process=4, sizes=(3000,1500)):
        
        func = partial(_get_member_scores, X_matrices, X_matrices_rc, y)

        print('starting')
        nucleotides = ['A', 'C', 'G', 'T']
        keywords = itertools.product(nucleotides, repeat = self.motif_length)
        kmer_list = ["".join(x) for x in keywords]

        cov = cov_init 
        size = sizes[-1]
        best_memory = None
        best_score = 10000000

        for i in range(iterations):
            print('iteration:', i)
            print('drawing samples')
            if i==0:
                full_grid = np.array([motif_to_beta(x) for x in kmer_list]) / 6
                members = full_grid[np.random.choice(range(len(full_grid)), size=sizes[0], replace=False)]
            else:
                members = multivariate_normal.rvs(mean=mu, cov=cov, size=size)

            print('calculating scores for samples')
            member_scores = np.array([self.loss_func(return_counts(y, classify_sequences(X_matrices, X_matrices_rc, member))) for member in members])
            #with Pool(num_process) as p:
            #    member_scores = np.array(p.map(func, members))

            ##Get the top samples##
            print('getting best scores')
            best_scoring_indices = np.argsort(member_scores)[0:20]
            if member_scores[best_scoring_indices[0]] < best_score:
                best_score = member_scores[best_scoring_indices[0]]
                best_memory = members[best_scoring_indices[0]]
            else:
                pass
            print('best score so far', best_score)
            print(member_scores[best_scoring_indices])

            ## Calculate the MLE ##
            print('calculating MLE')
            new_mu = np.mean(members[best_scoring_indices], axis=0)
            new_cov = np.mean([np.outer(x,x) for x in (members[best_scoring_indices] - new_mu)], axis=0)


            print("updating values")
            if i==0:
                mu = new_mu
                cov = alpha*new_cov + (1-alpha)*cov

            else:
                mu = alpha*new_mu + (1-alpha)*mu
                #if i == int(iterations/2)-1:
                #    cov=0.15
                #else:
                #    cov = alpha*new_cov + (1-alpha)*cov
                cov = alpha*new_cov + (1-alpha)*cov

            entropy = self.loss_func(return_counts(y, classify_sequences(X_matrices, X_matrices_rc, mu)))

            ##keeping track##
            self.beta_memory.append(mu)
            self.loss_memory.append(entropy)
            self.beta = mu

        if self.loss_func(return_counts(y, classify_sequences(X_matrices, X_matrices_rc, self.beta))) > best_score:
            print('going with something different!')
            self.beta = best_memory
        else:
            print('nah we good')
            
### everything here and below copied from fit
        classification = classify_sequences(X_matrices, X_matrices_rc, self.beta)
        print("counts...", return_counts(y, classification))
        current_entropy = self.loss_func(return_counts(y, classification))
        print("current entropy...", current_entropy)


        ## ONCE FIT, final classification of resulting nodes are defined ##
        final_counts = return_counts(y, classification)
        if final_counts[0] > final_counts[1]:
            self.left_classification = 1
        else:
            self.left_classification = 0

        if final_counts[2] > final_counts[3]:
            self.right_classification = 0
        else:
            self.right_classification = 1

        self.left_child_output = final_counts[0] / (final_counts[0] + final_counts[1])
        self.right_child_output = final_counts[3] / (final_counts[2] + final_counts[3])



    def split_points(self, indices, X_matrices, X_matrices_rc):

        classification = classify_sequences(X_matrices, X_matrices_rc, self.beta)

        left_split = indices[np.where(classification==1)[0]]
        right_split = indices[np.where(classification==0)[0]]

        return (left_split, right_split)

    def predict_one(self, x):
        if classify_sequence(x, self.beta, self.motif_length, self.seq_length) == 1:
            return self.left_classification
        else:
            return self.right_classification

    #def predict(self, X, X_rc):
    #    decision_function_output = decision_function(X, X_rc)
    #    g = lambda x: 1 if x>0.5 else 0
    #    return np.array([g(x) for x in decision_function_output])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def decision_function_one(self, x):
        if classify_sequence(x, self.beta, self.motif_length, self.seq_length) == 1:
            return self.left_child_output 
        else:
            return self.right_child_output


    def decision_function(self, X):
        return np.array([self.decision_function_one(x) for x in X])



    #def decision_function(self, X, X_rc):

    #    classified = classify_sequences(X, X_rc, self.beta)
    #    g = lambda x: self.left_child_output if x==1 else self.right_child_output
    #    return np.array([g(x) for x in classified])



class ConvDecisionTree:

    def __init__(self, depth, motif_length, seq_length):#, initial_betas, initial_beta_probabilities):
        self.depth = depth
        self.motif_length = motif_length
        self.seq_length = seq_length
        #self.initial_betas = initial_betas
        #self.initial_beta_probabilities = initial_beta_probabilities
        self.nodes = []

    def gradientfit(self, X, y, weights, iterations, step_size):
        data = []

        X_matrices = np.array([x_to_matrix(x, self.motif_length, self.seq_length) for x in np.array(X)])

        for layer in range(self.depth):
            #First layer go!
            if layer == 0:
                #node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                        beta0=motif_to_beta('CCCCCC'))
                node0.fit(X_matrices, y, weights, iterations, step_size)
                #node0.anneal(X_matrices, y, weights, alpha=0.9, T_start=.0005, T_min=0.0001, iterT=200)

                self.nodes.append([node0])
                data.append([node0.split_points(np.arange(len(X_matrices)), X_matrices)])

            #Rest of the layers
        else:

            #loop through the nodes from previous layer
                for i in range(len(self.nodes[layer-1])):

                    ### do this stuff only if the node was not terminal ###
                    if self.nodes[layer-1][i].terminal == False:

                        left = data[layer-1][i][0]
                        right = data[layer-1][i][1]

                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                                beta0=motif_to_beta('CCCCCC'))
                        temp_node_L.fit(X_matrices.take(left, axis=0), y.take(left), weights.take(left), iterations, step_size)
                        #temp_node_L.anneal(X_matrices.take(left, axis=0), y.take(left), weights.take(left), alpha=.9, T_start=.0005, T_min=.0001, iterT=200)

                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                                beta0=motif_to_beta('CCCCCC'))
                        temp_node_R.fit(X_matrices.take(right, axis=0), y.take(right), weights.take(right), iterations, step_size)
                        #temp_node_R.anneal(X_matrices.take(right, axis=0), y.take(right), weights.take(right), alpha=.9, T_start=.0005, T_min=.0001, iterT=200)

                        left_children = temp_node_L.split_points(left, X_matrices.take(left, axis=0))
                        right_children = temp_node_R.split_points(right, X_matrices.take(right, axis=0))


                        ######################################################################
                        #### Call it a terminal node if the child nodes don't have enough ####
                        ######################################################################
                        ##if (np.min([len(left_children[0]), len(left_children[1])]) < .00005*len(X_matrices)):
                        #if (len(left_children[0]) + len(left_children[1])) < .0000005*len(X_matrices):
                        #    temp_node_L.set_terminal_status(status=True)
                        #else:
                        #    pass

                        ##if (np.min([len(right_children[0]), len(right_children[1])]) < .00005*len(X_matrices)):
                        #if (len(right_children[0]) + len(right_children[1])) < .0000005*len(X_matrices):
                        #    temp_node_R.set_terminal_status(status=True)
                        #else:
                        #    pass

                        ######################################
                        ### Add the nodes and data to list ###
                        ######################################
                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([left_children, right_children])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([left_children, right_children])


                    else: #make dummy nodes and set status to terminal also
                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))

                        temp_node_L.set_terminal_status(status=True)
                        temp_node_R.set_terminal_status(status=True)



                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([data[layer-1][i], data[layer-1][i]])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([data[layer-1][i], data[layer-1][i]])

        for node in  self.nodes[-1]:
            node.set_terminal_status(status=True)



    def annealfit(self, X, y, weights, alpha=0.9, T_start=.0005, T_min=.0001, iterations=250):
        data = []

        X_matrices = np.array([x_to_matrix(x, self.motif_length, self.seq_length) for x in np.array(X)])

        for layer in range(self.depth):
            #First layer go!
            if layer == 0:
                #node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                #        beta0=motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1))
                node0 = ParaNode(motif_length=self.motif_length, seq_length=self.seq_length, 
                        beta_vec0=[motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1) for i in range(3)])

                node0.anneal(X_matrices, y, weights, alpha=alpha, T_start=T_start, T_min=T_min, iterT=iterations)

                self.nodes.append([node0])
                data.append([node0.split_points(np.arange(len(X_matrices)), X_matrices)])

            #Rest of the layers
        else:

            #loop through the nodes from previous layer
                for i in range(len(self.nodes[layer-1])):

                    ### do this stuff only if the node was not terminal ###
                    if self.nodes[layer-1][i].terminal == False:

                        left = data[layer-1][i][0]
                        right = data[layer-1][i][1]

                        #temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                        #        beta0=motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1))
                        temp_node_L = ParaNode(motif_length=self.motif_length, seq_length=self.seq_length, 
                                beta_vec0=[motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1) for i in range(3)])
                        temp_node_L.anneal(X_matrices.take(left, axis=0), y.take(left), weights.take(left), alpha=alpha, T_start=T_start, T_min=T_min, iterT=iterations)

                        #temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                        #        beta0=motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1))
                        temp_node_R = ParaNode(motif_length=self.motif_length, seq_length=self.seq_length, 
                                beta_vec0=[motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1) for i in range(3)])
                        temp_node_R.anneal(X_matrices.take(right, axis=0), y.take(right), weights.take(right), alpha=alpha, T_start=T_start, T_min=T_min, iterT=iterations)

                        left_children = temp_node_L.split_points(left, X_matrices.take(left, axis=0))
                        right_children = temp_node_R.split_points(right, X_matrices.take(right, axis=0))


                        ######################################################################
                        #### Call it a terminal node if the child nodes don't have enough ####
                        ######################################################################
                        if (np.min([len(left_children[0]), len(left_children[1])]) < .05*len(X_matrices)):
                            temp_node_L.set_terminal_status(status=True)
                        else:
                            pass

                        if (np.min([len(right_children[0]), len(right_children[1])]) < .05*len(X_matrices)):
                            temp_node_R.set_terminal_status(status=True)
                        else:
                            pass

                        ######################################
                        ### Add the nodes and data to list ###
                        ######################################
                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([left_children, right_children])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([left_children, right_children])


                    else: #make dummy nodes and set status to terminal also
                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))

                        temp_node_L.set_terminal_status(status=True)
                        temp_node_R.set_terminal_status(status=True)

                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([data[layer-1][i], data[layer-1][i]])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([data[layer-1][i], data[layer-1][i]])

        for node in  self.nodes[-1]:
            node.set_terminal_status(status=True)

    def crossentropyfit(self, X, X_rc, y, weights, iterations, num_process):
        data = []

        X_matrices = np.array([x_to_matrix(x, self.motif_length, self.seq_length) for x in np.array(X)])
        X_matrices_rc = np.array([x_to_matrix(x, self.motif_length, self.seq_length) for x in np.array(X_rc)])

        for layer in range(self.depth):
            #First layer go!
            if layer == 0:
                #node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                        beta0=motif_to_beta('CCCCCC'))
                node0.crossentropyFit(X_matrices, X_matrices_rc, y, weights, alpha=0.9, lam=0, cov_init=0.5, iterations=iterations, num_process=num_process)

                self.nodes.append([node0])
                data.append([node0.split_points(np.arange(len(X_matrices)), X_matrices, X_matrices_rc)])

            #Rest of the layers
            else:

            #loop through the nodes from previous layer
                for i in range(len(self.nodes[layer-1])):

                    ### do this stuff only if the node was not terminal ###
                    if self.nodes[layer-1][i].terminal == False:

                        left = data[layer-1][i][0]
                        right = data[layer-1][i][1]

                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                                beta0=motif_to_beta('CCCCCC'))
                        temp_node_L.crossentropyFit(X_matrices.take(left, axis=0), X_matrices_rc.take(left, axis=0), y.take(left), weights.take(left), alpha=0.9, lam=0, cov_init=0.5, 
                                iterations=iterations, num_process=num_process)

                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                                beta0=motif_to_beta('CCCCCC'))
                        temp_node_R.crossentropyFit(X_matrices.take(right, axis=0), X_matrices_rc.take(right, axis=0), y.take(right), weights.take(right), alpha=0.9, lam=0, cov_init=0.5, 
                                iterations=iterations, num_process=num_process)

                        left_children = temp_node_L.split_points(left, X_matrices.take(left, axis=0), X_matrices_rc.take(left, axis=0))
                        right_children = temp_node_R.split_points(right, X_matrices.take(right, axis=0), X_matrices_rc.take(right, axis=0))


                        ######################################################################
                        #### Call it a terminal node if the child nodes don't have enough ####
                        ######################################################################
                        #if (np.min([len(left_children[0]), len(left_children[1])]) < .05*len(X_matrices)):
                        #    temp_node_L.set_terminal_status(status=True)
                        #else:
                        #    pass

                        #if (np.min([len(right_children[0]), len(right_children[1])]) < .05*len(X_matrices)):
                        #    temp_node_R.set_terminal_status(status=True)
                        #else:
                        #    pass

                        ######################################
                        ### Add the nodes and data to list ###
                        ######################################
                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([left_children, right_children])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([left_children, right_children])


                    else: #make dummy nodes and set status to terminal also
                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))

                        temp_node_L.set_terminal_status(status=True)
                        temp_node_R.set_terminal_status(status=True)



                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([data[layer-1][i], data[layer-1][i]])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([data[layer-1][i], data[layer-1][i]])

        for node in  self.nodes[-1]:
            node.set_terminal_status(status=True)


    #def decision_function():



    def predict_one(self, x):

        #start with the head node
        current_layer = 0
        leftright = 0
        current_node = self.nodes[current_layer][leftright]

        terminal_node = False
        #loop uniil at terminal node
        while terminal_node == False:

            out = classify_sequence(x, current_node.beta, current_node.motif_length, current_node.seq_length)
            if out == 1:  ## if we get a hit, go to left node
                current_layer += 1
                leftright = leftright*2
                current_node = self.nodes[current_layer][leftright]

            else:  ## if no hit, go to right node
                current_layer += 1
                leftright = leftright*2 + 1
                current_node = self.nodes[current_layer][leftright]

            terminal_node = current_node.terminal

        return current_node.predict_one(x)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def decision_function_one(self, x):
        #start with the head node
        current_layer = 0
        leftright = 0
        current_node = self.nodes[current_layer][leftright]

        terminal_node = False
        #loop uniil at terminal node
        while terminal_node == False:

            out = classify_sequence(x, current_node.beta, current_node.motif_length, current_node.seq_length)
            if out == 1:  ## if we get a hit, go to left node
                current_layer += 1
                leftright = leftright*2
                current_node = self.nodes[current_layer][leftright]

            else:  ## if no hit, go to right node
                current_layer += 1
                leftright = leftright*2 + 1
                current_node = self.nodes[current_layer][leftright]

            terminal_node = current_node.terminal

        return current_node.decision_function_one(x)

    def decision_function(self, X):
        return np.array([self.decision_function_one(x) for x in X])



from sklearn.base import BaseEstimator, ClassifierMixin
class ConvDT(BaseEstimator):
    def __init__(self, depth, motif_length, sequence_length, iterations=10, num_processes=4, alpha=0.80, loss_function=two_class_weighted_entropy):
        self.depth = depth
        self.motif_length = motif_length
        self.sequence_length = sequence_length
        self.iterations = iterations
        self.alpha = alpha
        self.num_processes = num_processes
        self.loss_function = loss_function
        self.data = []
                

    def _find_optimal_beta(self, X_matrices, X_matrices_rc, y, weights, grid, cov_init=0.4, sizes=(3000,1500), elite_num=20):
        func = partial(_get_member_scores, X_matrices, X_matrices_rc, y, self.classes_, weights)

        cov = cov_init
        best_memory = None
        best_score = 1000000

        for i in range(self.iterations):
            print('iteration:', i)
            print('drawing samples')
            if i==0:
                members = grid[np.random.choice(range(len(grid)), size=sizes[0], replace=False)]
            else:
                members = multivariate_normal.rvs(mean=mu, cov=cov, size=sizes[1])

            print('calculating scores...')
            with Pool(self.num_processes) as p:
                member_scores = np.array(p.map(func, members))

            print('getting bets scores')
            best_scoring_indices = np.argsort(member_scores)[0:elite_num]
            if member_scores[best_scoring_indices[0]] < best_score:
                best_score = member_scores[best_scoring_indices[0]]
                best_memory = members[best_scoring_indices[0]]
            else:
                pass

            print('best score so far:', best_score)
            print(member_scores[best_scoring_indices])

            ## Calculate the MLE ##
            new_mu = np.mean(members[best_scoring_indices], axis=0)
            new_cov = np.mean([np.outer(x,x) for x in (members[best_scoring_indices] - new_mu)], axis=0) ## maybe faster way

            print('updating values')
            if i==0:
                mu = new_mu
                cov = self.alpha*new_cov + (1-self.alpha)*cov
            else:
                mu = self.alpha*new_mu + (1-self.alpha)*mu
                cov = self.alpha*new_cov + (1-self.alpha)*cov

            #entropy = self.loss_func(return_counts(y, classify_sequences(X_matrices, X_matrices_rc, mu)))

        #final_counts = 
       


        #if self.loss_function(return_counts_general(y, classify_sequences(X_matrices, X_matrices_rc, mu), self.classes_)) > best_score:
        if self.loss_function(return_counts_weighted(y, classify_sequences(X_matrices, X_matrices_rc, mu), self.classes_, weights)) > best_score:
            print('going with something else')
            return best_memory
        else:
            print('nah we good')
            return mu


    def _split_points(self, indices, X_matrices, X_matrices_rc, beta):
        classification = classify_sequences(X_matrices, X_matrices_rc, beta)

        left_split = indices[np.where(classification==1)[0]]
        right_split = indices[np.where(classification==0)[0]]

        return (left_split, right_split)

    

    def fit(self, X, y, sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones(len(X))

        self.data = []
        self.classes_ = np.unique(y)
        self.betas = []
        self.proportions = []

        #create X_matrices and its reverse complement
        X_matrices = np.array([x_to_matrix(x, self.motif_length, self.sequence_length) for x in np.array(X)])
        X_rc = np.array([x[::-1] for x in X])
        X_matrices_rc = np.array([x_to_matrix(x, self.motif_length, self.sequence_length) for x in np.array(X_rc)])


        print('creating grid')
        nucleotides = ['A', 'C', 'G', 'T']
        keywords = itertools.product(nucleotides, repeat=self.motif_length)
        kmer_list = ["".join(x) for x in keywords]
        full_grid = np.array([motif_to_beta(x) for x in kmer_list]) / 6.5 ### NEED A BETTER WAY TO DO THIS!!!

        for layer in range(self.depth):
            if layer == 0:
                self.betas.append([self._find_optimal_beta(X_matrices, X_matrices_rc, y, sample_weight, full_grid)])
                self.data.append([self._split_points(np.arange(len(X_matrices)), X_matrices, X_matrices_rc, self.betas[layer][0])])

                print('counts...', return_counts_weighted(y, classify_sequences(X_matrices, X_matrices_rc, self.betas[0][0]), self.classes_, sample_weight))

            else:
                for i in range(len(self.betas[layer-1])):
                    left = self.data[layer-1][i][0]
                    right = self.data[layer-1][i][1]

                    left_beta = self._find_optimal_beta(X_matrices.take(left, axis=0), X_matrices_rc.take(left, axis=0), y.take(left), sample_weight.take(left), full_grid)
                    print ('counts...', return_counts_weighted(y.take(left), classify_sequences(X_matrices.take(left, axis=0), X_matrices_rc.take(left, axis=0), left_beta), self.classes_, sample_weight.take(left)))
                    
                    right_beta = self._find_optimal_beta(X_matrices.take(right, axis=0), X_matrices_rc.take(right, axis=0), y.take(right), sample_weight.take(right), full_grid)
                    print ('counts...', return_counts_weighted(y.take(right), classify_sequences(X_matrices.take(right, axis=0), X_matrices_rc.take(right, axis=0), right_beta), self.classes_, sample_weight.take(right)))

                    left_children = self._split_points(left, X_matrices.take(left, axis=0), X_matrices_rc.take(left, axis=0), left_beta)
                    right_children = self._split_points(right, X_matrices.take(right, axis=0), X_matrices_rc.take(right, axis=0), right_beta)



                    if i==0: #have to append instead of extend on first iteration
                        self.betas.append([left_beta, right_beta])
                        self.data.append([left_children, right_children])
                    else:
                        self.betas[layer].extend([left_beta, right_beta])
                        self.data[layer].extend([left_children, right_children])


        for i in range(len(self.betas[-1])):
            left = self.data[-1][i][0]
            right = self.data[-1][i][1]
            left_proportion = (y.take(left) == self.classes_[0]).sum()/len(left)
            right_proportion = (y.take(right) == self.classes_[0]).sum()/len(right)
            self.proportions.extend([(left_proportion, 1-left_proportion), (right_proportion, 1-right_proportion)])
        
        return self
        

    def predict_proba_one(self, x):
        current_layer = 0
        position = 0

        for current_layer in range(self.depth):
            out = classify_sequence(x, self.betas[current_layer][position], self.motif_length, self.sequence_length)
            if out==1:#self.classes_[0]: ## if motif found, go to left node
                position = position*2
            else:
                position = position*2+1

        return self.proportions[position]

    def predict_proba(self, X):
        return np.array([self.predict_proba_one(x) for x in X])

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



    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(y)






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










