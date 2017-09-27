####################
## Beta functions ##
####################
def random_change(Beta, std=0.5):
    return [b+np.random.normal(scale=std) for b in Beta]

def small_change(Beta, std=0.2):
    length = int(len(Beta)/4)
    random_base = np.random.choice(range(length))
    new_Beta = copy.deepcopy(Beta)
    new_Beta[random_base*4:random_base*4+4] = [b + np.random.normal(scale=std) for b in new_Beta[random_base*4:random_base*4+4]]
    return new_Beta


def acceptable_beta(Beta, thresh):
    stride = Beta[0].itemsize
    Beta_ndarray = as_strided(Beta, shape=(6,4), strides=[stride*4,stride])
    return all(np.sum(np.abs(Beta_ndarray), axis=1) < thresh)

def get_new_beta(Beta, update_func, thresh, *args):
    while True:
        new_Beta = update_func(Beta, *args)
        #print(Beta)
        #print("TRYING!")
        if acceptable_beta(new_Beta, thresh):
            break
    return new_Beta

def get_new_betas(Beta_vec, update_func, thresh, *args):
    return [get_new_beta(b, update_func, thresh, *args) for b in Beta_vec]
    
# function that creates random initial beta
def random_beta(motif_length):
    output = []
    for i in range(motif_length):
        temp = np.zeros(4)
        num = np.random.choice(range(4))
        temp[num] = np.random.normal(loc=1/(motif_length-1), scale=0.01)
        output.extend(temp)

    return output





#######################################
##### GRADIENT DESCENT FUNCTIONS ######
#######################################
def single_sigmoid(x, alpha=100, offset=0.9):
    return 1/(1 + np.exp(-alpha*(x-offset)))

@vectorize('float64(float64, float64, float64)')
def single_sigmoid_vectorized(x, alpha=100, offset=0.9):
    return 1/(1 + np.exp(-alpha*(x-offset)))

@vectorize('float64(float64)')
def simplified_sigmoid(x):
    return 1/(1 + x)

def single_sigmoid_deriv(x, alpha=100, offset=0.9):
    exponent = np.exp(-alpha*(x-offset))
    return (alpha * exponent) / (1 + exponent)**2

@vectorize('float64(float64, float64, float64)')
def single_sigmoid_deriv_vectorized(x, alpha=100, offset=0.9):
    exponent = np.exp(-alpha*(x-offset))
    return (alpha * exponent) / (1 + exponent)**2

@vectorize('float64(float64, float64)')
def simplified_sigmoid_deriv(x, alpha):
    return (alpha * x)/((1 + x)**2)




### returns the sum of all the sigmoids of all subsequences
def sum_sigmoid_sequence(xdotbeta, motif_length, sequence_length):

    #x_matrix = x_to_matrix(x, motif_length, sequence_length)
    vectorized_single_sigmoid = np.vectorize(single_sigmoid)

    return np.sum(vectorized_single_sigmoid(xdotbeta, alpha=100, offset=0.9))

@jit
def better_sum_sigmoid_sequence(x, beta, motif_length, sequence_length):
    x_matrix = x_to_matrix(x, motif_length, sequence_length)

    output = 0
    for m in np.dot(x_matrix, beta):
        output += m

    return output



def sum_sigmoid_deriv_sequence(x, beta, motif_length, sequence_length):

    x_matrix = x_to_matrix(x, motif_length, sequence_length)
    vectorized_single_sigmoid_deriv = np.vectorize(single_sigmoid_deriv)    

    return np.sum(np.dot(np.diag(vectorized_single_sigmoid_deriv(np.dot(x_matrix, beta), alpha=100, offset=0.9)), x_matrix), 
            axis=0)



def gradient(X, y, beta, motif_length, sequence_length):

    X_positive = X[y==1]
    X_negative = X[y==0]


    A = [1, 0, 0, 0]
    C = [0, 1, 0, 0]
    G = [0, 0, 1, 0]
    T = [0, 0, 0, 1]
    nucleotides = [A, C, G, T]

    combinations = [[item for sublist in p for item in sublist] for p in itertools.product(nucleotides, repeat=6)]
    combinations = ["".join([str(x) for x in combo]) for combo in combinations]

    combination_lookupvalues = np.exp(-100 * (np.dot(combinations, beta) - 0.9))

    lookuptable = dict(zip(combinations, combination_lookupvalues))


    total = []

    p = np.sum([single_sigmoid(sum_sigmoid_sequence(lookuptable[x_to_string(X.ix[i])], motif_length, sequence_length)) for i in y[y==1].index.values])
    n = np.sum([single_sigmoid(sum_sigmoid_sequence(lookuptable[x_to_string(X.ix[i])], beta, motif_length, sequence_length)) for i in y[y==0].index.values])

    P = len(y[y==1])
    N = len(y[y==0])

    print(p, n, P-p, N-n)

    p_factor = (y==0).apply(lambda x: int(x))*(np.log(n/(N-n))-np.log((p+n)/(P+N-p-n)))+(y==1).apply(lambda x: int(x))*(np.log(p/(P-p))-np.log((p+n)/(P+N-p-n)))


    for x in np.array(X):
        S = sum_sigmoid_sequence(x, beta, motif_length, sequence_length)
        first_term = single_sigmoid_deriv(S)
        second_term = sum_sigmoid_deriv_sequence(x, beta, motif_length, sequence_length)

        total.append(first_term * second_term)

    first = entropy([P, N])
    second = two_class_weighted_entropy([p, n, N-n, P-p])

    output = np.dot(np.array(p_factor), np.array(total))
    return output/(np.sum(np.abs(output))*8) , (first - second)

def newnewgradient(X_matrices, y, beta, motif_length, sequence_length, step_size=1/50):

    #X_matrices = [x_to_matrix(x, motif_length, sequence_length) for x in np.array(X)]
    a = np.array([np.dot(x, beta) for x in X_matrices])
    sig_sum = [np.sum(single_sigmoid_vectorized(x, 100, 0.9)) for x in a]
    b = [single_sigmoid_deriv_vectorized(x, 100, 0.9) for x in a]
    c = [np.sum(X_matrices[i] * b[i][:,np.newaxis], axis=0) for i in range(len(X_matrices))]
    d = [single_sigmoid_deriv(x) for x in sig_sum]

    p = pd.Series(sig_sum)[(y==1).as_matrix()].apply(single_sigmoid, args=(100,0.9)).sum()
    n = pd.Series(sig_sum)[(y==0).as_matrix()].apply(single_sigmoid, args=(100,0.9)).sum()

    P = len(y[y==1])
    N = len(y[y==0])

    p_factor = (y==0).apply(lambda x: int(x))*(np.log(n/(N-n))-np.log((p+n)/(P+N-p-n)))+(y==1).apply(lambda x: int(x))*(np.log(p/(P-p))-np.log((p+n)/(P+N-p-n)))

    gradient = np.sum((c * np.array(d)[:, np.newaxis]) * p_factor[:, np.newaxis], axis=0)

    return (gradient/(np.sqrt(np.dot(gradient,gradient)) * (1/step_size))), [p, n, N-n, P-p]

def weightedgradient(X_matrices, y, weights, beta, motif_length, sequence_length, step_size=1/50):
    weights_series = pd.Series(weights)

    #X_matrices = [x_to_matrix(x, motif_length, sequence_length) for x in np.array(X)]
    a = np.array([np.dot(x, beta) for x in X_matrices])
    sig_sum = [np.sum(single_sigmoid_vectorized(x, 100, 0.9)) for x in a]
    b = [single_sigmoid_deriv_vectorized(x, 100, 0.9) for x in a]
    c = [np.sum(X_matrices[i] * b[i][:,np.newaxis], axis=0) for i in range(len(X_matrices))]
    d = [single_sigmoid_deriv(x) for x in sig_sum] * weights
    #print(len(d))

    p = (pd.Series(sig_sum)[(y==1)].apply(single_sigmoid, args=(100,0.9)) * weights[y==1]).sum()
    n = (pd.Series(sig_sum)[(y==0)].apply(single_sigmoid, args=(100,0.9)) * weights[y==0]).sum()

    #print(p, n)

    P = weights_series[y==1].sum()
    N = weights_series[y==0].sum()

    #p_factor = (y==0).apply(lambda x: int(x))*(np.log(n/(N-n))-np.log((p+n)/(P+N-p-n)))+(y==1).apply(lambda x: int(x))*(np.log(p/(P-p))-np.log((p+n)/(P+N-p-n)))
    p_factor = (y==0)*(np.log(n/(N-n))-np.log((p+n)/(P+N-p-n)))+(y==1)*(np.log(p/(P-p))-np.log((p+n)/(P+N-p-n)))

    gradient = np.sum((c * np.array(d)[:, np.newaxis]) * p_factor[:, np.newaxis], axis=0)

    #return (gradient/np.sum(np.abs(gradient) * 15)), [p, n, N-n, P-p]
    return (gradient/(np.sqrt(np.dot(gradient,gradient)) * (1/step_size))), [p, n, N-n, P-p]


def Information_Gain(X, y, beta, motif_length, sequence_length):

    p = np.sum([single_sigmoid(sum_sigmoid_sequence(X.ix[i], beta, motif_length, sequence_length)) for i in y[y==1].index.values])
    n = np.sum([single_sigmoid(sum_sigmoid_sequence(X.ix[i], beta, motif_length, sequence_length)) for i in y[y==0].index.values])

    P = len(y[y==1])
    N = len(y[y==0])

    first = entropy([P, N])
    second = two_class_weighted_entropy([p, n, P-p, N-n])

    return first - second


def acceptance_probability(initial, final, T):
    return np.exp((initial - final)/T)
