import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

PLOT_FLAG = True


def generate_MoG_data(num_data, means, covariances, weights):
    """
    Generate a list of data points from given mixture of Gaussians model.
    :param num_data: number of data points
    :param means: means of Gaussians
    :param covariances: covariance of Gaussians
    :param weights: weights of each model
    :return: a list of data points
    """
    num_clusters = len(weights)
    data = []
    for i in range(num_data):
        #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
        k = np.random.choice(len(weights), 1, p=weights)[0]

        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covariances[k])

        data.append(x)
    return data

# A simple mixture of models:
print("Generate data points from a simple mixture of models:")
# Model parameters
init_means = [
    [5, 0], # mean of cluster 1
    [1, 1], # mean of cluster 2
    [0, 5]  # mean of cluster 3
]
init_covariances = [
    [[.5, 0.], [0, .5]],         # covariance of cluster 1
    [[.92, .38], [.38, .91]],    # covariance of cluster 2
    [[.5, 0.], [0, .5]]          # covariance of cluster 3
]
init_weights = [1/4., 1/2., 1/4.]  # weights of each cluster

# Generate data
np.random.seed(4)
data = generate_MoG_data(100, init_means, init_covariances, init_weights)

# Report
print("    Initial means: " + str(init_means))
print("    Initial covariances: " + str(init_covariances))
print("    Initial Weights: " + str(init_weights))

# Plot
if PLOT_FLAG:
    plt.figure()
    d = np.vstack(data)
    plt.plot(d[:,0], d[:,1],'ko')
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()


if PLOT_FLAG:
    plt.show()