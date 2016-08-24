import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import os
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

PLOT_FLAG = False


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


def get_initial_centroids(data, k, seed=None):
    """
    Randomly choose k data points as initial centroids
    :param data: whole dataset
    :param k: number of initial centroids
    :param seed: random seed
    :return: k randomly chosen initial centroids
    """

    if seed is not None:  # useful for obtaining consistent results
        np.random.seed(seed)
    n = data.shape[0]  # number of data points

    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)

    # Keep centroids as dense format, as many entries will be nonzero due to averaging.
    # As long as at least one document in a cluster contains a word,
    # it will carry a nonzero weight in the TF-IDF vector of the centroid.
    centroids = data[rand_indices, :].toarray()
    return centroids


def assign_clusters(data, centroids):
    """
    Assign closest centroid to each data point.
    :param data: whole dataset
    :param centroids: given centroids
    :return: array of centroids index indicating the closest centroid
    """
    # Compute distances between each data point and the set of centroids:
    distances_from_centroids = pairwise_distances(data, centroids, metric='euclidean')
    # Compute cluster assignments for each data point:
    cluster_assignment = np.argmin(distances_from_centroids, axis=1)
    return cluster_assignment


def revise_centroids(data, k, cluster_assignment):
    """
    Revise centroids by taking mean values of data points assigned to each centroid.
    :param data: whole dataset
    :param k: number of centroids
    :param cluster_assignment: indices of centroid each data point assigned to
    :return: new centroids
    """
    new_centroids = []
    for i in range(k):
        # Select all data points that belong to cluster i.
        member_data_points = data[cluster_assignment == i, :]
        # Compute the mean of the data points.
        centroid = np.mean(member_data_points, axis=0)

        # Convert numpy.matrix type to numpy.ndarray type
        centroid = centroid.A1
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)

    return new_centroids


def compute_heterogeneity(data, k, centroids, cluster_assignment):
    """
    Compute the sum of all squared distances between data points and assigned centroids.
    The smaller the distances, the more homogeneous the clusters are.
    :param data: whole dataset
    :param k: number of centroids
    :param centroids: all centroids
    :param cluster_assignment: indices of centroid data points assigned to
    :return: distance to test heterogeneity
    """
    heterogeneity = 0.0
    for i in range(k):

        # Select all data points that belong to cluster i.
        member_data_points = data[cluster_assignment == i, :]

        if member_data_points.shape[0] > 0:  # check if i-th cluster is non-empty
            # Compute distances from centroid to data points
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances ** 2
            heterogeneity += np.sum(squared_distances)

    return heterogeneity


def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    """
    Runs k-means on given data and initial set of centroids.
    :param data: whole dataset
    :param k: number of centroids
    :param initial_centroids: initial centroids
    :param maxiter: maximum number of iterations to run
    :param record_heterogeneity: a list, to store the history of heterogeneity as function of iterations
                                 if None, do not store the history.
    :param verbose: if True, print how many data points changed their cluster labels in each iteration
    :return: final centroids and assignment indices
    """

    centroids = initial_centroids[:]
    prev_cluster_assignment = None

    for itr in range(maxiter):
        # Make cluster assignments using nearest centroids
        cluster_assignment = assign_clusters(data, centroids)

        # Compute a new centroid for each of the k clusters,
        # averaging all data points assigned to that cluster.
        centroids = revise_centroids(data, k, cluster_assignment)

        # Check for convergence: if none of the assignments changed, stop
        if prev_cluster_assignment is not None and \
           (prev_cluster_assignment == cluster_assignment).all():
            break

        # Print number of new assignments
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment != cluster_assignment)
            if verbose:
                print('Iter {:3d}: {:5d} elements changed their cluster assignment.'
                      .format(itr, num_changed))

        # Record heterogeneity convergence metric
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)

        prev_cluster_assignment = cluster_assignment[:]

    return centroids, cluster_assignment


def plot_heterogeneity(heterogeneity, k):
    """
    Plot heterogeneity distance.
    :param heterogeneity: heterogeneity distance array
    :param k: number of centroids
    :return:
    """
    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# Load data
wiki = pd.read_csv("../Data/people_wiki.csv")
wiki['id'] = wiki.index
tf_idf = load_sparse_csr("../Data/people_wiki_tf_idf.npz")
tf_idf = normalize(tf_idf)
with open("../Data/people_wiki_map_index_to_word.json") as json_data:
    map_index_to_word = json.load(json_data)

# Look at documents 100 through 102 as query documents
print("Look at documents 100 through 102 as query documents:")
# Get the TF-IDF vectors for documents 100 through 102.
queries = tf_idf[100:102, :]

# Compute pairwise distances from every data point to each query vector.
dist = pairwise_distances(tf_idf, queries, metric='euclidean')
print("The distance from every data point to each query vector:")
print(dist)
print()

# Initialize three centroids with the first 3 rows of tf_idf.
# Write code to compute distances from each of the centroids to all data points in tf_idf.
# Then find the distance between row 430 of tf_idf and the second centroid and save it to dist.
print("Take first 3 rows of tf_idf as centroids.")
print("The distance from all data points to each of the centroids:")
centroids = tf_idf[:3, :]
dist = pairwise_distances(tf_idf, centroids, metric='euclidean')
print(dist)

# Test
if np.allclose(dist[430, 1], pairwise_distances(tf_idf[430, :], tf_idf[1, :])):
    print('Test Pass!')
else:
    print('Check your code again!')
print()

# Take the minimum of the distances for each data point
closest_cluster = np.argmin(dist, axis=1)
print("The centroid indices that each data point assigned:")
print(closest_cluster)

# Test
reference = [list(row).index(min(row)) for row in dist]
if np.allclose(closest_cluster, reference):
    print('Test Pass!')
else:
    print('Check your code again!')
print()

# Testing assign_clusters
print("Testing function assign_clusters...")
if np.allclose(assign_clusters(tf_idf[0:100:10], tf_idf[0:8:2]), np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1])):
    print('Test Pass!')
else:
    print('Check your code again!')
print()

# Testing revise_centroids
print("Testing function revise_centroids...")
result = revise_centroids(tf_idf[0:100:10], 3, np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1]))
if np.allclose(result[0], np.mean(tf_idf[[0, 30, 40, 60]].toarray(), axis=0)) and \
   np.allclose(result[1], np.mean(tf_idf[[10, 20, 90]].toarray(), axis=0)) and \
   np.allclose(result[2], np.mean(tf_idf[[50, 70, 80]].toarray(), axis=0)):
    print('Test Pass!')
else:
    print('Check your code again!')
print()

# Run k-means with k=3 on TF-IDF dataset
print("Run k-means with k=3 on TF-IDF dataset:")
k = 3
heterogeneity = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                       record_heterogeneity=heterogeneity, verbose=True)
print()

# Report
num_of_assignments = np.bincount(cluster_assignment)
print("{:15s}{:5s}{:30s}".format("Index of Cluster", "", "# of Data Points Assigned"))
for i in range(k):
    print("{:8d}{:5s}{:20d}".format(i, "", num_of_assignments[i]))
print()

if PLOT_FLAG:
    plot_heterogeneity(heterogeneity, k)

# QUIZ QUESTIONS:
print("Quiz Questions:")
# 1. (True/False) The clustering objective (heterogeneity) is non-increasing for this example.
print("1. The clustering objective (heterogeneity) is non-increasing for this example: {:s}.\n"
      .format(str(np.all(np.diff(heterogeneity) <= 0))))
# 2. If the clustering objective (heterogeneity) would ever increase when running k-means,
#    that would indicate: (choose one)
#        k-means algorithm got stuck in a bad local minimum
#        There is a bug in the k-means code
#        All data points consist of exact duplicates
#        Nothing is wrong. The objective should generally go down sooner or later.
print("2. If the clustering objective (heterogeneity) would ever increase when running k-means,"
      " that would indicate THERE IS A BUG IN THE K-MEANS CODE.\n")
# 3. Which of the cluster contains the greatest number of data points in the end?
print("3. Cluster #{:d} contains the greatest number of data points.\n"
      .format(np.argmax(num_of_assignments)))

if PLOT_FLAG:
    plt.show()

