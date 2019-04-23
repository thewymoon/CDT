
import numpy as np

### loss functions ###

def child_variance(classifications,y_values,weights):
    classes = np.unique(classifications)
    total = 0 
    for unique in classes:
        temp_indices = np.where(classifications==unique)[0]
        total += len(temp_indices)*np.var(y_values[temp_indices])
    return total


def child_entropy(classifications,y,weights,pseudo=0.01, classes=[0,1]):
    counts = []
            
    class_indices = []
    for class_value in classes:
        class_indices.append(np.where(y==class_value)[0])

    for i in range(2):
        counts.extend([np.sum(weights[np.where(classifications[indices]==i)[0]]) for indices in class_indices])

    return (my_entropy([counts[0], counts[1]], pseudo=pseudo)*np.sum(counts[0:2]) + my_entropy([counts[2], counts[3]], pseudo=pseudo)*np.sum(counts[2:4]))/np.sum(counts)

def my_entropy(p_vec, pseudo=0.01):
    if np.sum(p_vec) > 0:
        return np.sum([-(p)*np.log((p)) for p in [(x/np.sum(p_vec))+pseudo for x in p_vec]])
    else:
        return 0
