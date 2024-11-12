import numpy as np
import math
import scipy as scy
import scipy.optimize as scyopt
from sympy import gamma, Float
from numba import jit,njit,prange,vectorize,float64
import pyFCI


################################################################################
@njit(parallel=True,fastmath=True)
def center_and_normalize(dataset):
    """
    Center and normalize a **dataset** of N d-dimensional points so that its Euclidean barycenter is the zero vector, and each of its points has norm 1. 

    :param dataset: vector of shape (N,d)
    :returns: vector of shape (N,d)
    """
    (n1,n2) = np.shape(dataset)
    r = np.empty((n1,n2))
    mean = np.empty(n2)
    for i in prange(n2):
        mean[i] = np.mean(dataset[:,i])
    for i in prange(n1):
        v = dataset[i] - mean
        r[i] = v / np.linalg.norm(v)
    return r

################################################################################
@njit(parallel=True,fastmath=True)
def FCI(dataset):
    """
    Compute the full correlation integral of a **dataset** of N d-dimensional points by exact enumeration

    :param dataset: vector of shape (N,d)
    :returns: vector of shape (N(N-1)/2,2)
    """
    n = len(dataset)
    m = int(n*(n-1)/2)
    rs = np.empty(m)
    for i in prange(n):
        for j in prange(i+1,n):
            c = int( -0.5 * i *  (1 + i - 2 * n) + (j - i) - 1 )
            rs[c] = np.linalg.norm(dataset[i]-dataset[j]) 
    rs = np.sort(rs)
    r = np.empty((m,2))
    for i in prange(m):
        r[i] = np.array([ rs[i] , i*1./m ])
    return r

################################################################################
@njit(parallel=True, fastmath=True)
def FCI_MC(dataset, n_samples=500):
    """
    Compute the full correlation integral of a dataset by randomly sampling pairs of points.

    This function calculates the distances between randomly selected pairs of points 
    in the dataset and returns a sorted array representing the full correlation integral.

    :param dataset: A numpy array of shape (N, d) where N is the number of points and 
                    d is the dimensionality of each point.
    :param n_samples: An integer representing the number of point pairs to sample.
                      Defaults to 500.
    :returns fci: A numpy array of shape (n_samples, 2) containing sorted distances and their 
              corresponding cumulative distribution values.
    """
    n = len(dataset)  # Number of points in the dataset
    m = int(n * (n - 1) / 2)  # Total possible pairs in the dataset
    n_samples = min(m, n_samples)  # Ensure samples do not exceed possible pairs
    sample_distances = np.empty(n_samples)  # Array to store sampled distances
    
    # Randomly sample indices for selecting pairs
    random_indices = np.random.choice(m, n_samples, replace=False)
    
    # Compute distances for randomly selected pairs
    for k, index in enumerate(random_indices):
        i = math.floor((2 * n - 1 - math.sqrt((2 * n - 1)**2 - 8 * index)) / 2)
        j = index - (i * (2 * n - i - 1)) // 2 + i + 1
        sample_distances[k] = np.linalg.norm(dataset[i] - dataset[j])
    
    # Sort distances to form the correlation integral
    sample_distances = np.sort(sample_distances)
    
    # Prepare the output array with sorted distances and their distribution
    fci = np.empty((n_samples, 2))
    for i in prange(n_samples):
        fci[i] = np.array([sample_distances[i], i * 1. / n_samples])
    
    return fci
"""
random_pairs = np.random.choice(m1,samples,replacement=False)

rs = np.empty(samples)

for k in prange(samples):

    index=random_pairs[k]

    i = math.floor((2 * n - 1 - math.sqrt((2 * n - 1)**2 - 8 * index)) / 2)
    j = index - (i * (2 * n - i - 1)) // 2 + i + 1
"""

################################################################################
@jit(forceobj=True,fastmath=True)
def analytical_FCI(x,d,x0=1):
    """
    Compute the analytical average full correlation integral on a **d**-dimensional sphere at **x**

    :param x: a real number in (0,2), or a vector of real numbers in (0,2)
    :param d: a real positive number
    :param x0: a real number (should be close to 1). It's such that f(x0)=0.5
    :returns: a real number, or a numpy vector of real numbers
    """
    return  0.5 * ( 1 + float(Float((gamma((1+d)/2)) / (np.sqrt(np.pi) * gamma(d/2) ))) * (-2+(x/x0)**2) * scy.special.hyp2f1( 0.5, 1-d/2, 3/2, 1/4 * (-2+(x/x0)**2)**2 ) )

################################################################################
@jit(forceobj=True,fastmath=True)
def fit_FCI(rho, samples=500, threshold=0.1):
    """
    Given an empirical full correlation integral **rho**, it tries to fit it to the analytical_FCI curve.
    To avoid slow-downs, only a random sample of **samples** points is used in the fitting.
    If the fit fails, it outputs [0,0,0]

    :param rho: vector of shape (N,2) of points in (0,2)x(0,1)
    :param samples: a positive integer
    :returns: the fitted dimension, the fitted x0 parameter and the mean square error between the fitted curve and the empirical points
    """
    samples = min( len(rho),samples )
    data = rho[np.random.choice(len(rho),samples,replace=False)]

    fit = scyopt.curve_fit( analytical_FCI, data[:,0], data[:,1] )
    if abs(fit[0][1] - 1)>threshold:
        return [0,0,0]
    else:
        mse = np.sqrt(np.mean([ (pt[1] - analytical_FCI(pt[0],fit[0][0],fit[0][1]))**2 for pt in data ]))
        return [fit[0][0]+1,fit[0][1],mse]

################################################################################

# TODO: modify to have also a cutoff by distance, and not only by kNN 

def local_FCI(dataset, center, ks):
    """
    Given a **dataset** of N d-dimensional points, the index **center** of one of the points and a list of possible neighbourhoods **ks**, it estimates the local intrinsic dimension by using **fit_FCI()** of the reduced dataset of the first k-nearest-neighbours of **dataset[center]**, for each k in **ks**

    At the moment, it uses FCI_MC and fit FCI with default parameters

    :param dataset: a vector of shape (N,d)
    :param center: the index of a point in **dataset**
    :param ks: list of increasing positive integers
    :returns: a vector of shape (len(ks),5). For each k in **ks**, returns the list [ k, distance between dataset[center] and the k-th neighbour, fitted dimension, fitted x0, the mean square error of the fit ] 
    """
    neighbours = dataset[np.argsort(np.linalg.norm( dataset - dataset[center], axis=1))[0:ks[-1]]]    
  
    local = np.empty(shape=(0,5))
    for k in ks:
        fit = fit_FCI( FCI_MC( center_and_normalize( neighbours[0:k] ) ) )
        local = np.append(local, [[ k, np.linalg.norm( neighbours[k-1] - neighbours[0] ), fit[0], fit[1], fit[2] ]], axis=0 )

    return local


