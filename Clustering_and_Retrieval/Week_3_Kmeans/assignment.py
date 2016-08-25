import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
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
    plt.figure(figsize=(7, 4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


def smart_initialize(data, k, seed=None):
    """
    Use k-means++ to initialize a good set of centroids
    :param data: whole dataset
    :param k: number of centroids
    :param seed: random seed
    :return: initial centroids
    """
    if seed is not None:  # useful for obtaining consistent results
        np.random.seed(seed)
    centroids = np.zeros((k, data.shape[1]))

    # Randomly choose the first centroid.
    # Since we have no prior knowledge, choose uniformly at random
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx, :].toarray()
    # Compute distances from the first centroid chosen to all the other data points
    distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()

    for i in range(1, k):
        # Choose the next centroid randomly, so that the probability for each data point to be chosen
        # is directly proportional to its squared distance from the nearest centroid.
        # Roughly speaking, a new centroid should be as far as from other centroids as possible.
        idx = np.random.choice(data.shape[0], 1, p=distances / sum(distances))
        centroids[i] = data[idx, :].toarray()
        # Now compute distances from the centroids to all data points
        distances = np.min(pairwise_distances(data, centroids[0:i + 1], metric='euclidean'), axis=1)

    return centroids


def kmeans_multiple_runs(data, k, maxiter, num_runs, seed_list=None, verbose=False):
    """
    Run k-means for multiple times. using k-means++ to initialize centroids.
    :param data: whole dataset
    :param k: number of centroids
    :param maxiter: maximum number of iterations to run
    :param num_runs: number of k-means to run
    :param seed_list: list of random seeds
    :param verbose: if True, print how many data points changed their cluster labels in each iteration
    :return: final centroids and assignment indices
    """
    heterogeneity = {}

    min_heterogeneity_achieved = float('inf')
    best_seed = None
    final_centroids = None
    final_cluster_assignment = None

    for i in range(num_runs):

        # Use UTC time if no seeds are provided
        if seed_list is not None:
            seed = seed_list[i]
            np.random.seed(seed)
        else:
            seed = int(time.time())
            np.random.seed(seed)

        # Use k-means++ initialization
        initial_centroids = smart_initialize(data, k, seed)

        # Run k-means
        centroids, cluster_assignment = kmeans(data, k, initial_centroids, maxiter,
                                               record_heterogeneity=None, verbose=False)

        # To save time, compute heterogeneity only once in the end
        heterogeneity[seed] = compute_heterogeneity(data, k, centroids, cluster_assignment)

        if verbose:
            print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
            sys.stdout.flush()

        # if current measurement of heterogeneity is lower than previously seen,
        # update the minimum record of heterogeneity.
        if heterogeneity[seed] < min_heterogeneity_achieved:
            min_heterogeneity_achieved = heterogeneity[seed]
            best_seed = seed
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment

    # Return the centroids and cluster assignments that minimize heterogeneity.
    return final_centroids, final_cluster_assignment


def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    """
    Plot k vs heterogeneity distance.
    :param k_values: number of centroids
    :param heterogeneity_values: heterogeneity distance
    :return: None
    """
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


def visualize_document_clusters(wiki, tf_idf, centroids, cluster_assignment, k, map_index_to_word_df,
                                display_content=True):
    """
    Visualize clustering by printing titles, first sentences, and top 5 words with highest TF-IDF score.
    :param wiki: original wiki dataset
    :param tf_idf: tf-idf dataset
    :param centroids: clustering centroids
    :param cluster_assignment: indices of centroids of data points assigned to
    :param k: number of centroids
    :param map_index_to_word_df: Dataframe contains mapping between words and column indices
    :param display_content: if True, display 8 nearest neighbors of each centroid
    :return: None
    """
    print('==========================================================')

    # Visualize each cluster c
    for c in range(k):
        # Cluster heading
        print('Cluster {0:d}    '.format(c)),
        # Print top 5 words with largest TF-IDF weights in the cluster
        idx = centroids[c].argsort()[::-1]
        for i in range(5):  # Print each word along with the TF-IDF weight
            word = map_index_to_word_df['word'][map_index_to_word_df['index'] == idx[i]].values[0]
            print('{0:s}: {1:.3f}'.format(word, centroids[c, idx[i]]), end=' ')
        print()

        if display_content:
            # Compute distances from the centroid to all data points in the cluster,
            # and compute nearest neighbors of the centroids within the cluster.
            distances = pairwise_distances(tf_idf, [centroids[c]], metric='euclidean').flatten()
            distances[cluster_assignment != c] = float('inf')  # remove non-members from consideration
            nearest_neighbors = distances.argsort()
            # For 8 nearest neighbors, print the title as well as first 180 characters of text.
            # Wrap the text at 80-character mark.
            for i in range(8):
                target_line = wiki[wiki['id'] == nearest_neighbors[i]]
                text = ' '.join(target_line['text'].values[0].split(None, 25)[0:25])
                print('\n* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(target_line['name'].values[0],
                                                                     distances[nearest_neighbors[i]], text[:90],
                                                                     text[90:180] if len(text) > 90 else ''))
        print('==========================================================')

# Load data
wiki = pd.read_csv("../Data/people_wiki.csv")
wiki['id'] = wiki.index
tf_idf = load_sparse_csr("../Data/people_wiki_tf_idf.npz")
tf_idf = normalize(tf_idf)
with open("../Data/people_wiki_map_index_to_word.json") as json_data:
    map_index_to_word = json.load(json_data)
map_index_to_word_df = pd.DataFrame({
    'index': pd.Series(list(map_index_to_word.values())),
    'word': pd.Series(list(map_index_to_word.keys()))
})

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
heterogeneity_1 = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                       record_heterogeneity=heterogeneity_1, verbose=True)
print()

# Report
num_of_assignments = np.bincount(cluster_assignment)
print("{:15s}{:5s}{:30s}".format("Index of Cluster", "", "# of Data Points Assigned"))
for i in range(k):
    print("{:8d}{:5s}{:20d}".format(i, "", num_of_assignments[i]))
print()

if PLOT_FLAG:
    plot_heterogeneity(heterogeneity_1, 3)

# Run k-means multiple times, with different initial centroids created using different random seeds.
print("Run k-means multiple times with different randomly chosen initial centroids:")
k = 10
heterogeneity_normal = {}
largest_cluster_sizes_normal = {}
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = get_initial_centroids(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity_normal[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    largest_cluster_sizes_normal[seed] = np.max(np.bincount(cluster_assignment))
    print('seed={0:06d}, heterogeneity={1:.5f}, largest cluster size={2:d}'
          .format(seed, heterogeneity_normal[seed], largest_cluster_sizes_normal[seed]))
    sys.stdout.flush()
end = time.time()
print("Running time: {:.4f} seconds\n".format(end-start))

# Run k-means multiple times, with k-means++ to choose initial centroids.
print("Run k-means multiple times with different smart chosen initial centroids:")
k = 10
heterogeneity_smart = {}
largest_cluster_sizes_smart = {}
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = smart_initialize(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    largest_cluster_sizes_smart[seed] = np.max(np.bincount(cluster_assignment))
    print('seed={0:06d}, heterogeneity={1:.5f}, largest cluster size={2:d}'
          .format(seed, heterogeneity_smart[seed], largest_cluster_sizes_smart[seed]))
    sys.stdout.flush()
end = time.time()
print("Running time: {:.4f} seconds\n".format(end-start))

# Plot
if PLOT_FLAG:
    plt.figure(figsize=(8, 5))
    plt.boxplot([list(heterogeneity_normal.values()), list(heterogeneity_smart.values())], vert=False)
    plt.yticks([1, 2], ['k-means', 'k-means++'])
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# Choose k
print("Run k-means with different k (number of centroids):")
'''
Following Codes Runs for about ONE HOUR. Use pre-computed values instead.

# start = time.time()
# centroids = {}
# cluster_assignment = {}
# heterogeneity_values = []
# k_list = [2, 10, 25, 50, 100]
# seed_list = [0, 20000, 40000, 60000, 80000, 100000, 120000]

# for k in k_list:
#     heterogeneity = []
#     centroids[k], cluster_assignment[k] = kmeans_multiple_runs(tf_idf, k, maxiter=400,
#                                                                num_runs=len(seed_list),
#                                                                seed_list=seed_list,
#                                                                verbose=True)
#     score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
#     heterogeneity_values.append(score)
#     print("k={:d}, heterogeneity={:f}".format(k, score))

# if PLOT_FLAG:
#     plot_k_vs_heterogeneity(k_list, heterogeneity_values)

#end = time.time()
#print("Running time: {:.4f} seconds\n".format(end-start))
'''

filename = '../Data/kmeans-arrays.npz'

heterogeneity_var_k_values = []
k_list = [2, 10, 25, 50, 100]

if os.path.exists(filename):
    arrays = np.load(filename)
    centroids = {}
    cluster_assignment = {}
    for k in k_list:
        centroids[k] = arrays['centroids_{0:d}'.format(k)]
        cluster_assignment[k] = arrays['cluster_assignment_{0:d}'.format(k)]
        score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
        heterogeneity_var_k_values.append(score)
        print("k={:d}, heterogeneity={:f}".format(k, score))
    print()

    if PLOT_FLAG:
        plot_k_vs_heterogeneity(k_list, heterogeneity_var_k_values)

else:
    print('File not found. Skipping.')

# Visualize document clusters
print("Visualize document clusters:")
print("k = 2:")
visualize_document_clusters(wiki, tf_idf, centroids[2], cluster_assignment[2], 2, map_index_to_word_df)
print('''Cluster 0: artists, songwriters, professors, politicians, writers, etc.
Cluster 1: baseball players, hockey players, football (soccer) players, etc.\n''')

print("k = 10:")
visualize_document_clusters(wiki, tf_idf, centroids[10], cluster_assignment[10],
                            10, map_index_to_word_df, display_content=False)
print('''Cluster 0: artists, poets, writers, environmentalists
Cluster 1: film directors
Cluster 2: female figures from various fields
Cluster 3: politicians
Cluster 4: track and field athletes
Cluster 5: composers, songwriters, singers, music producers
Cluster 6: soccer (football) players
Cluster 7: baseball players
Cluster 8: professors, researchers, scholars
Cluster 9: lawyers, judges, legal scholars\n''')

print("k = 25:")
visualize_document_clusters(wiki, tf_idf, centroids[25], cluster_assignment[25],
                            25, map_index_to_word_df, display_content=False)
print('''Cluster 0: composers, songwriters, singers, music producers
Cluster 1: poets
Cluster 2: rugby players
Cluster 3: baseball players
Cluster 4: government officials
Cluster 5: football players
Cluster 6: radio hosts
Cluster 7: actors, TV directors
Cluster 8: professors, researchers, scholars
Cluster 9: lawyers, judges, legal scholars
Cluster 10: track and field athletes
Cluster 11: (mixed; no clear theme)
Cluster 12: car racers
Cluster 13: priets, bishops, church leaders
Cluster 14: painters, sculptors, artists
Cluster 15: novelists
Cluster 16: American football players
Cluster 17: golfers
Cluster 18: American politicians
Cluster 19: basketball players
Cluster 20: generals of U.S. Air Force
Cluster 21: politicians
Cluster 22: female figures of various fields
Cluster 23: film directors
Cluster 24: music directors, composers, conductors\n''')

print("k = 100:")
visualize_document_clusters(wiki, tf_idf, centroids[100], cluster_assignment[100],
                            100, map_index_to_word_df, display_content=False)
print('''When k = 100, the class of rugby players have been broken into two clusters (11 and 72).
Same goes for soccer (football) players (clusters 6, 21, 40, and 87),
although some may like the benefit of having a separate category for Australian Football League.
The class of baseball players have been also broken into two clusters (18 and 95).\n''')

# QUIZ QUESTIONS:
print("Quiz Questions:")
# 1. (True/False) The clustering objective (heterogeneity) is non-increasing for this example.
print("1. The clustering objective (heterogeneity) is non-increasing for this example: {:s}.\n"
      .format(str(np.all(np.diff(heterogeneity_1) <= 0))))
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
# 4. Look at the size of the largest cluster (most # of member data points) across multiple runs.
#    How much does this measure vary across the runs?
#    What is the minimum and maximum values this quantity takes?
largest_cluster_sizes_normal_values = list(largest_cluster_sizes_normal.values())
print("4. The size of the largest cluster across multiple runs using random initialization are:")
print("   " + str(largest_cluster_sizes_normal_values))
print("   The minimum value is {:d}. The maximum value is {:d}.\n"
      .format(np.min(largest_cluster_sizes_normal_values),
              np.max(largest_cluster_sizes_normal_values)))
# 5. Take k = 10, which of the 10 clusters above contains the greatest number of articles?
#    Which of the 10 clusters contains the least number of articles?
articles_count_10 = np.bincount(cluster_assignment[10])
print("5. When k = 10: Cluster {:d} contains the greatest number of articles. "
      "Cluster {:d} contains the least number of articles.\n"
      .format(np.argmax(articles_count_10), np.argmin(articles_count_10)))
# 6. Another sign of too large K is having lots of small clusters.
#    Look at the distribution of cluster sizes (by number of member data points).
#    How many of the 100 clusters have fewer than 236 articles, i.e. 0.4% of the dataset?
articles_count_100 = np.bincount(cluster_assignment[100])
print("6. When k = 100, there are {:d} clusters have fewer than 236 articles, i.e. 0.4% of the dataset."
      .format(np.size(np.argwhere(articles_count_100 < 236))))

if PLOT_FLAG:
    plt.show()