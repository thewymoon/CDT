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

#import nltk


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
    #return np.logical_or(np.any(np.dot(X_matrices, beta) > 1, axis=1), np.any(np.dot(X_matrices_rc, beta) > 1, axis=1)).astype(int)
    #return np.logical_or(np.any(np.array([np.dot(x, beta) for x in X_matrices]) > 1, axis=1), np.any(np.array([np.dot(x, beta) for x in X_matrices_rc]) > 1, axis=1)).astype(int)
    return np.logical_or(np.any(faster_dot(X_matrices, beta) >= 1.0, axis=1), np.any(faster_dot(X_matrices_rc, beta) >= 1.0, axis=1)).astype(int)

def weighted_classify_sequences(X_matrices, X_matrices_rc, beta, weights):
    #return np.logical_or(np.any(np.dot(X_matrices, beta) > 1, axis=1), np.any(np.dot(X_matrices_rc, beta) > 1, axis=1)).astype(int)
    #return np.logical_or(np.any(np.array([np.dot(x, beta) for x in X_matrices]) > 1, axis=1), np.any(np.array([np.dot(x, beta) for x in X_matrices_rc]) > 1, axis=1)).astype(int)
    return np.logical_or(np.any(faster_dot(X_matrices, beta) >= 1.0, axis=1), np.any(faster_dot(X_matrices_rc, beta) >= 1.0, axis=1)).astype(int)
    
def classify_sequence(x, beta, motif_length, sequence_length):
    x_matrix = x_to_matrix(x, motif_length, sequence_length)
    return int(np.any(np.dot(x_matrix, beta) >= 1.0) or np.any(np.dot(x_matrix, flip_beta(beta)) >= 1.0))

    


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



## Functions for determining initial grid of cross entroyp ##
def calculate_prob(N, l, x):
    return 1 - (1 - (scipy.misc.comb(l,x) * 3**(l-x) * (1/4)**l))**(2*(N-l+1))

def find_best_div(N, l, proportion):
    best = 1
    closest = 1
    for d in np.arange(0,l,.1):
        diff = np.abs((calculate_prob(N, l, d) - proportion))
        if diff < closest:
            closest = diff
            best = d
        else:
            pass
    return best



def x_to_string(x):
    return "".join([str(i) for i in x])


def x_to_matrix(x, motif_length, sequence_length):
    numpy_arrayx = np.array(x)
    size = numpy_arrayx.itemsize

    #print('size', size)
    return as_strided(numpy_arrayx, shape = [sequence_length - motif_length, motif_length*4], strides = [size*4,size])


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



from sklearn.base import BaseEstimator, ClassifierMixin
class ConvDT(BaseEstimator):
    def __init__(self, depth, motif_length, sequence_length, iterations=10, num_processes=4, alpha=0.80, loss_function=two_class_weighted_entropy, optimization_sample_size=(3000,1500)):
        self.depth = depth
        self.motif_length = motif_length
        self.sequence_length = sequence_length
        self.iterations = iterations
        self.alpha = alpha
        self.num_processes = num_processes
        self.loss_function = loss_function
        self.data = []
        self.optimization_sample_size = optimization_sample_size
                

    def _find_optimal_beta(self, X_matrices, X_matrices_rc, y, weights, grid, cov_init=0.4, elite_num=20):
        func = partial(_get_member_scores, X_matrices, X_matrices_rc, y, self.classes_, weights)

        cov = cov_init
        best_memory = None
        best_score = 1000000

        for i in range(self.iterations):
            print('iteration:', i)
            print('drawing samples')
            if i==0:
                members = grid[np.random.choice(range(len(grid)), size=self.optimization_sample_size[0], replace=False)]
            else:
                members = multivariate_normal.rvs(mean=mu, cov=cov, size=self.optimization_sample_size[1])

            print('calculating scores...')
            with Pool(self.num_processes) as p:
                member_scores = np.array(p.map(func, members))

            print('getting best scores')
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
        self.n_classes_ = len(self.classes_)
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
        #div = find_best_div(self.sequence_length, self.motif_length, 0.5)
        #print(div)
        full_grid = np.array([motif_to_beta(x) for x in kmer_list]) / 6.5

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
                    print ('counts...', return_counts_weighted(y.take(left), 
                        classify_sequences(X_matrices.take(left, axis=0), X_matrices_rc.take(left, axis=0), left_beta), self.classes_, sample_weight.take(left)))
                    
                    right_beta = self._find_optimal_beta(X_matrices.take(right, axis=0), X_matrices_rc.take(right, axis=0), y.take(right), sample_weight.take(right), full_grid)
                    print ('counts...', return_counts_weighted(y.take(right), 
                        classify_sequences(X_matrices.take(right, axis=0), X_matrices_rc.take(right, axis=0), right_beta), self.classes_, sample_weight.take(right)))

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










