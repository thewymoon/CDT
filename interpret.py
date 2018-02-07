import numpy as np
from ConvDT import flip_beta, x_to_matrix

## returns list of indices/locations on sequence x where we get a hit on beta
def find_where(x, beta):
    motif_length = int(len(beta)/4)
    sequence_length = int(len(x)/4)

    betaRC = flip_beta(beta)
    x_matrix = x_to_matrix(x, motif_length=motif_length, sequence_length=sequence_length)
    
    
    outputs = np.dot(x_matrix, beta)
    outputsRC = np.dot(x_matrix, betaRC)
    
    result = []

    for i in range(len(outputs)):
        if outputs[i] >= 1:
            result.append(i)
    for j in range(len(outputsRC)):
        if outputsRC[j] >= 1:
            result.append(j)
    
    return result

## takes x with dummy features and returns the sequence string
def x_to_seq(x):
    motif_length = int(len(x)/4)
    nucleotides = ['A', 'C', 'G', 'T']
    
    return ''.join([nucleotides[i] for i in np.where(x.reshape(motif_length, 4) == 1)[-1]])

## returns list of the sequences on x that get hit by beta
def get_hit_sequences(x, beta):
    hit_indices = find_where(x, beta)
    
    return [x_to_seq(x[i:i+len(beta)]) for i in hit_indices]

## returns a list of all betas in a boosted convDT
def get_all_betas(BDT):
    all_betas = []
    for tree in BDT.estimators_:
        for row in tree.betas:
            all_betas.extend(row)
    return all_betas

## returns list of the locations/indices on "sequence" that get hit by a beta in BDT
def get_hit_locations(sequence, BDT):
    locations = []
    for l in [find_where(sequence, b) for b in get_all_betas(BDT)]:
        if l:
            locations.extend(l)
    return locations


## returns a sequence without a window from idx to idx+length zeroed out
def censored_sequence(x, idx, length):
    return np.array([g if i not in range(idx*4,(idx+length)*4) else 0. for i,g in enumerate(x)])

## returns the 'importance scores' at each position on the sequence
def sequence_importances(sequence, BDT):
    sequence_length = int(len(sequence)/4)
    motif_length = BDT.base_estimator_.motif_length
    
    bdt_output = BDT.predict_proba([sequence])
    locations = get_hit_locations(sequence, BDT)
    if locations:
        differences = ([BDT.predict_proba([censored_sequence(sequence, i, motif_length)])[0] for i in locations] - bdt_output)[:,0]
    else:
        differences = []
#     print(differences)
#     print(locations)
    
    importances = np.zeros(sequence_length)
    for i,x in zip(locations, differences):
        importances[i:i+motif_length] += x
    
#     print(importances)
    return importances

##########################################
## Functions to return ranges of values ##
##########################################
def nonzero_ranges(values):
    ranges = []
    temp = []
    state = 0
    for i in range(len(values)):
        if state==0 and values[i]==0:
            pass
        elif state==0 and values[i]!=0:
            temp.append(i)
            state = 1
        elif state==1 and values[i]!=0:
            pass
        elif state==1 and values[i]==0:
            temp.append(i)
            state = 0
            ranges.append(temp)
            temp=[]
    return ranges

def positive_ranges(values):
    ranges = []
    temp = []
    state = 0
    for i in range(len(values)):
        if state==0 and values[i]<=0:
            pass
        elif state==0 and values[i]>0:
            temp.append(i)
            state = 1
        elif state==1 and values[i]>0:
            pass
        elif state==1 and values[i]<=0:
            temp.append(i)
            state = 0
            ranges.append(temp)
            temp=[]
    return ranges

def negative_ranges(values):
    ranges = []
    temp = []
    state = 0
    for i in range(len(values)):
        if state==0 and values[i]>=0:
            pass
        elif state==0 and values[i]<0:
            temp.append(i)
            state = 1
        elif state==1 and values[i]<0:
            pass
        elif state==1 and values[i]>=0:
            temp.append(i)
            state = 0
            ranges.append(temp)
            temp=[]
    return ranges


###############
## Mutations ##
###############
def crispr(sequence, location, base):
    nucleotides = {'A':[1.,0.,0.,0.], 'C':[0.,1.,0.,0.], 'G':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.], 'N':[0.,0.,0.,0.]}
    new = [x if i!= location else nucleotides[base] for i,x in enumerate(sequence.reshape(int(len(sequence)/4), 4))]
#     new[location] = nucleotides[base]
#     return new.flatten()
    return np.array(new).flatten()

def multi_crispr(sequence, locations, bases):
    
    current_sequence = sequence
    for i in range(len(locations)):
        current_sequence = crispr(current_sequence, locations[i], bases[i])
    
    return current_sequence
