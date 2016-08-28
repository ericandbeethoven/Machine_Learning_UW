import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.stats import multivariate_normal
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


def load_sparse_csr(filename):
    """
    Read a numpy data file (.npz) and return as a compressed sparse row matrix.
    :param filename: data file name
    :return: CSR matrix
    """
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)


def diag(array):
    """Create a sparse matrix with diagonal as the given array."""
    n = len(array)
    return spdiags(array, 0, n, n)


def logpdf_diagonal_gaussian(x, mean, cov):
    """
    Compute log-pdf of a multivariate Gaussian distribution with diagonal covariance at a given point x.
    A multivariate Gaussian distribution with a diagonal covariance is equivalent
    to a collection of independent Gaussian random variables.

    The log-pdf will be computed for each row of x.
    mean and cov should be given as 1D numpy arrays.

    :param x: a sparse matrix
    :param mean: means of variables
    :param cov: covariances of variables
    :return: log-pdf of a multivariate Gaussian distribution
    """
    n = x.shape[0]
    dim = x.shape[1]
    assert(dim == len(mean) and dim == len(cov))

    # multiply each i-th column of x by (1/(2*sigma_i)), where sigma_i is sqrt of variance of i-th variable.
    scaled_x = x.dot(diag(1. / (2 * np.sqrt(cov))))
    # multiply each i-th entry of mean by (1/(2*sigma_i))
    scaled_mean = mean / (2 * np.sqrt(cov))

    # sum of pairwise squared Eulidean distances gives SUM[(x_i - mean_i)^2/(2*sigma_i^2)]
    dist_sqr = pairwise_distances(scaled_x, [scaled_mean], 'euclidean').flatten() ** 2
    return -np.sum(np.log(np.sqrt(2 * np.pi * cov))) - dist_sqr


def log_sum_exp(x, axis):
    """Compute the log of a sum of exponentials"""
    x_max = np.max(x, axis=axis)
    if axis == 1:
        return x_max + np.log(np.sum(np.exp(x - x_max[:, np.newaxis]), axis=1))
    else:
        return x_max + np.log(np.sum(np.exp(x - x_max), axis=0))


def run_em_for_high_dimension(data, means, covs, weights, cov_smoothing=1e-5,
                              maxiter=int(1e3), thresh=1e-4, verbose=False):
    """
    Run EM algorithm on data to assign models.
    :param data: whole dataset
    :param means: initial means
    :param covariances: initial covariance matrices
    :param weights: initial list of weights of models
    :param cov_smoothing: specifies the default variance assigned to absent features in a cluster.
                          If we were to assign zero variances to absent features, we would be overconfient,
                          as we hastily conclude that those features would NEVER appear in the cluster.
                          We'd like to leave a little bit of possibility for absent features to show up later.
    :param maxiter: maximum number of iterations
    :param thresh: threshold on log-likelihood to stop algorithm
    :param verbose: print progress if True
    :return: {'weights': weights, 'means': means, 'covs': covariances,
              'loglik': log-likelihood trace, 'resp': responsibilities matrix}
    """
    n = data.shape[0]
    dim = data.shape[1]
    mu = deepcopy(means)
    Sigma = deepcopy(covs)
    K = len(mu)
    weights = np.array(weights)

    ll = None
    ll_trace = []

    for i in range(maxiter):
        # E-step: compute responsibilities
        logresp = np.zeros((n, K))
        for k in range(K):
            logresp[:, k] = np.log(weights[k]) + logpdf_diagonal_gaussian(data, mu[k], Sigma[k])
        ll_new = np.sum(log_sum_exp(logresp, axis=1))
        if verbose:
            print(ll_new)
        logresp -= np.vstack(log_sum_exp(logresp, axis=1))
        resp = np.exp(logresp)
        counts = np.sum(resp, axis=0)

        # M-step: update weights, means, covariances
        weights = counts / np.sum(counts)
        for k in range(K):
            mu[k] = (diag(resp[:, k]).dot(data)).sum(axis=0) / counts[k]
            mu[k] = mu[k].A1

            Sigma[k] = diag(resp[:, k]).dot(data.power(2) - 2 * data.dot(diag(mu[k]))).sum(axis=0) \
                       + (mu[k] ** 2) * counts[k]
            Sigma[k] = Sigma[k].A1 / counts[k] + cov_smoothing * np.ones(dim)

        # check for convergence in log-likelihood
        ll_trace.append(ll_new)
        if ll is not None and (ll_new - ll) < thresh and ll_new > -np.inf:
            ll = ll_new
            break
        else:
            ll = ll_new

    out = {'weights': weights, 'means': mu, 'covs': Sigma, 'loglik': ll_trace, 'resp': resp}
    return out


if __name__ == '__main__':
    # Load data
    # Take only first 5000 documents
    wiki = pd.read_csv("../Data/people_wiki.csv")
    wiki['id'] = wiki.index
    wiki = wiki.head(5000)
    tf_idf = load_sparse_csr("../Data/4_tf_idf.npz")
    tf_idf = normalize(tf_idf)
    with open("../Data/4_map_index_to_word.json") as json_data:
        map_index_to_word = json.load(json_data)

    # Initializing mean parameters using k-means
    print("Initializing mean parameters using k-means...")
    # Use scikit-learn's k-means to simplify workflow
    np.random.seed(5)
    num_clusters = 25
    kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1)
    kmeans_model.fit(tf_idf)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    means = [centroid for centroid in centroids]
    print("The cluster assignments after initialization by k-means are: " + str(cluster_assignment))
    print()

    # Initializing cluster weights to be the proportion of documents assigned to that cluster by k-means above.
    print("Initializing cluster weights...")
    num_docs = tf_idf.shape[0]
    weights = []
    for i in range(num_clusters):
        # Compute the number of data points assigned to cluster i:
        num_assigned = np.sum(cluster_assignment == i)
        w = float(num_assigned) / num_docs
        weights.append(w)
    print("The initial weights of clusters are: ")
    for i in range(num_clusters):
        print("\tCluster {:d}: {:.4f}".format(i, weights[i]))
    print()

