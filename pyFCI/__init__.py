import numpy as np
import math
import scipy as scy
import scipy.optimize as scyopt
from scipy.spatial.distance import pdist
from sympy import gamma, Float
from numba import jit,njit,prange,vectorize,float64
import pyFCI


################################################################################
def center_and_normalize(points):
    """
    Center and normalize a dataset of points.

    Parameters
    ----------
    points : array of shape (n_samples, d)
        Input data.

    Returns
    -------
    normalized_points : array of shape (n_samples, d)
        Centered and normalized points.
    """
    n_samples, d = points.shape
    mean = np.mean(points, axis=0)
    centered_points = points - mean

    norms = np.linalg.norm(centered_points, axis=1)
    normalized_points = centered_points / norms[:, np.newaxis]

    return normalized_points

################################################################################
def FCI(dataset):
    """
    Compute the full correlation integral of a **dataset** of N d-dimensional points by exact enumeration

    :param dataset: vector of shape (N,d)
    :returns: vector of shape (N(N-1)/2,2)
    """
    num_points = len(dataset)
    num_pairs = num_points * (num_points - 1) // 2
    pair_distances = pdist(dataset)
    sorted_distances = np.sort(pair_distances)
    correlation_integral = np.empty((num_pairs, 2))
    correlation_integral[:,0] = sorted_distances
    correlation_integral[:,1] = np.arange(0,num_pairs)/num_pairs
    return correlation_integral
################################################################################
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
    m = n * (n - 1) // 2  # Total possible pairs in the dataset

    if m < n_samples:
        sample_distances = pdist(dataset)
        n_samples = m
    else:
        n_samp = math.ceil((-1 + math.sqrt(1 + 8 * n_samples)) / 2) + 1
        random_indices = np.random.choice(n, n_samp, replace=False)
        sample_distances = pdist(dataset[random_indices])
        n_samples = n_samp * (n_samp - 1) // 2
    
    # Sort distances to form the correlation integral
    sample_distances = np.sort(sample_distances)
    
    # Prepare the output array with sorted distances and their distribution
    fci = np.empty((n_samples, 2))
    fci[:,0] = sample_distances
    fci[:,1] = np.arange(0,n_samples)/n_samples
    return fci
    
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
  
    local = np.empty(shape=(len(ks),5))

    for i, k in enumerate(ks):
        fit = fit_FCI( FCI_MC( center_and_normalize( neighbours[0:k] ) ) )
        local[i] = [ k, np.linalg.norm( neighbours[k-1] - neighbours[0] ), fit[0], fit[1], fit[2] ]

    return local

def main():
    import time
    x = np.random.randn(10000,2)
    y = np.zeros((10000,3))
    y[:,:2] = x
    y_norm = center_and_normalize(y)
    ti = time.time()
    fci = FCI_MC(y_norm)
    tf = time.time()
    fit = fit_FCI(fci)

    print(fit)
    print("time =",tf-ti)

    print("local_FCI")
    til = time.time()
    local = local_FCI(y, 0, range(4,100,10))
    tfl = time.time()
    print(local)
    print(tfl-til)

if __name__ == "__main__":
    main()