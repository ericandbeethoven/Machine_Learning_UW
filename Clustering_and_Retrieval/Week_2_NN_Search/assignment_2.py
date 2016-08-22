import numpy as np
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from copy import copy
from itertools import combinations

PLOT_FLAG = False


def norm(x):
    """
    Compute norm of a sparse vector.
    :param x: sparse vector
    :return: 2-norm of given sparse vector
    """
    sum_sq = x.dot(x.T)
    norm = np.sqrt(sum_sq)
    return norm


def cosine_distance(x, y):
    """
    Compute cosine distance of two vectors.
    :param x: vector x
    :param y: vector y
    :return: cosine distance of vector x and vector y
    """
    xy = x.dot(y.T)
    dist = xy / (norm(x) * norm(y))
    return 1 - dist[0, 0]


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


def generate_random_vectors(num_vector, dim):
    """
    Generate a collection of random vectors from the standard Gaussian distribution.
    :param num_vector: number of vectors will be generated     n
    :param dim: dimension of vectors                           d
    :return: generated random vectors                          n x d
    """
    return np.random.randn(dim, num_vector)


def brute_force_query(vec, data, k):
    """
    Run brute force search to find k nearest neighbors of query vector.
    :param vec: query vector
    :param data: dataset
    :param k: number of nearest neighbors
    :return: Dataframe of k nearest nearest neighbors
    """
    num_data_points = data.shape[0]

    # Compute distances for ALL data points in training set
    nearest_neighbors = pd.DataFrame({
        'id': pd.Series(range(num_data_points)),
        'distance': pd.Series(pairwise_distances(data, vec, metric='cosine').flatten())
    })

    return nearest_neighbors.sort_values('distance').head(k)


def train_lsh(data, num_vector=16, seed=None):
    """
    Train a Locality Sensitive Hashing model from given data
    :param data: data to perform LSH
    :param num_vector: hash size (bit representation)
    :param seed: random seed using when generate random vectors
    :return: LSH model
    """
    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

    table = {}
    # Partition data points into bins
    bin_index_bits = (data.dot(random_vectors) >= 0)

    # Encode bin index bits into integers
    bin_indices = bin_index_bits.dot(powers_of_two)

    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # If no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = []
        # Fetch the list of document ids associated with the bin and add the document id to the end.
        table[bin_index].append(data_index)

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}

    return model


def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    """
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.
    :param query_bin_bits: query bin vector
    :param table: table of bin vectors
    :param search_radius: max number of bits that are different from query bits vector
    :param initial_candidates: set that contains initial candidates
    :return: set of nearby bin vectors
    """
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

    # Allow the user to provide an initial set of candidates.
    candidate_set = copy(initial_candidates)

    for different_bits in combinations(range(num_vector), search_radius):
        # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
        # Hint: you can iterate over a tuple like a list
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = not alternate_bits[i]

        # Convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)

        # Fetch the list of documents belonging to the bin indexed by the new bit vector.
        # Then add those documents to candidate_set
        # Make sure that the bin exists in the table!
        # Hint: update() method for sets lets you add an entire list to the set
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])  # Update candidate_set with the documents in this bin.

    return candidate_set


def query(vec, model, k, max_search_radius):
    """
    Collect all candidates and compute their true distance to the query vector.
    Then return k nearest neighbors of query vector.
    :param vec: query vector
    :param model: LSH model
    :param k: number of nearest neighbors
    :param max_search_radius: max number of bits that are different from query bits vector
    :return: k nearest neighbors of query vector
             number of candidates searched
    """
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    num_vector = random_vectors.shape[1]

    # Compute bin index for the query vector, in bit representation.
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()

    # Search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)

    # Sort candidates by their true distances from the query
    candidates = data[np.array(list(candidate_set)), :]
    candidate_distance = pairwise_distances(candidates, vec, metric='cosine').flatten()
    nearest_neighbors = pd.DataFrame({
        'id': pd.Series(list(candidate_set)),
        'distance': pd.Series(candidate_distance)
    })

    return nearest_neighbors.sort_values('distance').head(k), len(candidate_set)

# Load data
wiki = pd.read_csv("../Data/people_wiki.csv")
wiki['id'] = wiki.index
corpus = load_sparse_csr("../Data/people_wiki_tf_idf.npz")
with open("../Data/people_wiki_map_index_to_word.json") as json_data:
    map_index_to_word = json.load(json_data)

# Generate random vectors of the same dimensionality as our vocabulary size.
# We generate 16 vectors, leading to a 16-bit encoding of the bin index for each document.
vocab_size = corpus.shape[1]         # Should be 547979
np.random.seed(0)
random_vectors = generate_random_vectors(num_vector=16, dim=vocab_size)

# Decide which bin document should go using 16 bits representation
# Each bit is given by the sign of dot product of random vector and document's TF-IDF vector.
print("Part 1: Putting data into bins")
powers_of_two = (1 << np.arange(15, -1, -1))

document_hash_bits_1 = corpus[0, :].dot(random_vectors) >= 0
print("The sign of dot product between random vectors and first document vector, 1 is positive:")
print(np.array(document_hash_bits_1, dtype=int))
print("The hash value of first document vector: {:d}\n"
      .format(document_hash_bits_1.dot(powers_of_two).tolist()[0]))

document_hash_bits_2 = corpus[1, :].dot(random_vectors) >= 0
print("The sign of dot product between random vectors and second document vector, 1 is positive:")
print(np.array(document_hash_bits_2, dtype=int))
print("The hash value of first document vector: {:d}\n"
      .format(document_hash_bits_2.dot(powers_of_two).tolist()[0]))

index_bits = corpus.dot(random_vectors) >= 0
print("The hash values of all documents:")
print(index_bits.dot(powers_of_two))
print()

# Check train_lsh
print("Checking train_lsh function:")
model = train_lsh(corpus, num_vector=16, seed=143)
table = model['table']
if 0 in table and table[0] == [39583] and \
   143 in table and table[143] == [19693, 28277, 29776, 30399]:
    print('Passed!')
else:
    print('Check your code.')
print()

# Part 2: Inspect Bins
print("Part 2: Inspect Bins")

# Compare Joe Biden and Wynn Normington Hugh-Jones with Obama
obama_id = wiki[wiki['name'] == 'Barack Obama'].index[0]
print("Barack Obama's article ID is {:d}.".format(obama_id))
obama_bin_bits = model['bin_index_bits'][obama_id]
obama_bin = model['bin_indices'][obama_id]
print("Barack Obama's article is in bin with index {:d}.".format(obama_bin))
print("The bits representation of Obama's bin index is:")
print(np.array(obama_bin_bits, dtype=int))
print()

biden_id = wiki[wiki['name'] == 'Joe Biden'].index[0]
print("Joe Biden's article ID is {:d}.".format(biden_id))
biden_bin_bits = model['bin_index_bits'][biden_id]
biden_bin = model['bin_indices'][biden_id]
print("Joe Biden's article is in bin with index {:d}.".format(biden_bin))
print("The bits representation of Biden's bin index is:")
print(np.array(biden_bin_bits, dtype=int))
print("Biden's bin representation agrees with Obama's in {:d} out of 16 places."
      .format(np.sum(obama_bin_bits == biden_bin_bits)))
print()

wynn_id = wiki[wiki['name'] == 'Wynn Normington Hugh-Jones'].index[0]
print("Wynn Normington Hugh-Jones's article ID is {:d}.".format(wynn_id))
wynn_bin_bits = model['bin_index_bits'][wynn_id]
wynn_bin = model['bin_indices'][wynn_id]
print("Wynn Normington Hugh-Jones's article is in bin with index {:d}.".format(wynn_bin))
print("The bits representation of Hugh-Jones's bin index is:")
print(np.array(wynn_bin_bits, dtype=int))
print("Hugh-Jones's bin representation agrees with Obama's in {:d} out of 16 places."
      .format(np.sum(obama_bin_bits == wynn_bin_bits)))
print()

# Get documents in the same bin as Obama
docs_in_obama_bin_ids = list(model['table'][model['bin_indices'][obama_id]])
docs_in_obama_bin_ids.remove(obama_id)        # display documents other than Obama

docs = wiki[wiki['id'].isin(docs_in_obama_bin_ids)]
print("Documents in the same bin as Obama's article:")
print(docs[['name']])
print()

# Compute cosine distances between Obama, with Biden and other persons in the same bin.
obama_tf_idf = corpus[obama_id, :]
biden_tf_idf = corpus[biden_id, :]

print('Cosine distance from Barack Obama:')
print('Barack Obama - {:24s}: {:f}'
      .format('Joe Biden', cosine_distance(obama_tf_idf, biden_tf_idf)))

for doc_id in docs_in_obama_bin_ids:
    doc_tf_idf = corpus[doc_id, :]
    print('Barack Obama - {:24s}: {:f}'
          .format(wiki['name'][doc_id], cosine_distance(obama_tf_idf, doc_tf_idf)))
print()

# Part 3: Query the LSH model
# 1. Decide on the search radius r. This will determine the number of different bits between the two vectors.
# 2. For each subset (n_1, n_2, ..., n_r) of the list [0, 1, 2, ..., num_vector-1], do the following:
#    * Flip the bits (n_1, n_2, ..., n_r) of the query bin to produce a new bit vector.
#    * Fetch the list of documents belonging to the bin indexed by the new bit vector.
#    * Add those documents to the candidate set.
print('Part 3: Query the LSH model')

# Check search_nearby_bins function
# Check search_radius = 0
print('Checking search_nearby_bins function:')
candidate_set = search_nearby_bins(obama_bin_bits, model['table'], search_radius=0)
if candidate_set == {35817, 21426, 53937, 39426, 50261}:
    print('Passed!')
else:
    print('Check your code.')

# Check search_radius = 1
candidate_set = search_nearby_bins(obama_bin_bits, model['table'], search_radius=1, initial_candidates=candidate_set)
if candidate_set == {39426, 38155, 38412, 28444, 9757, 41631, 39207, 59050, 47773, 53937, 21426, 34547,
                     23229, 55615, 39877, 27404, 33996, 21715, 50261, 21975, 33243, 58723, 35817, 45676,
                     19699, 2804, 20347}:
    print('Passed!')
else:
    print('Check your code.')
print()

nn_obama, len_nn_obama_candidates = query(corpus[obama_id, :], model, k=10, max_search_radius=3)
print("10 Nearest Neighbor of Obama's article:")
print(nn_obama.merge(wiki, on='id').sort_values('distance')[['id', 'distance', 'name']].to_string(index=False))
print()

# Part 4: Experimenting with LSH
print("Part 4: Experimenting with LSH")

# run LSH multiple times, each with different radii for nearby bin search
print("Run LSH with different search radii:")
num_candidates_history = []
query_time_history = []
max_distance_from_query_history = []
min_distance_from_query_history = []
average_distance_from_query_history = []

biden_min_radius = float('Inf')
for max_search_radius in range(17):
    start = time.time()
    # Perform LSH query using Barack Obama, with max_search_radius
    result, num_candidates = query(corpus[obama_id, :], model, k=10,
                                   max_search_radius=max_search_radius)
    end = time.time()
    query_time = end - start  # Measure time

    print('Radius:', max_search_radius)
    # Display 10 nearest neighbors, along with document ID and name
    print(result.merge(wiki, on='id').sort_values('distance')[['id', 'distance', 'name']].to_string(index=False))
    print()

    biden_found = not result[result['id'] == biden_id].empty
    if biden_found and max_search_radius < biden_min_radius:
        biden_min_radius = max_search_radius

    # Collect statistics on 10 nearest neighbors
    average_distance_from_query = result['distance'][1:].mean()
    max_distance_from_query = result['distance'][1:].max()
    min_distance_from_query = result['distance'][1:].min()

    num_candidates_history.append(num_candidates)
    query_time_history.append(query_time)
    average_distance_from_query_history.append(average_distance_from_query)
    max_distance_from_query_history.append(max_distance_from_query)
    min_distance_from_query_history.append(min_distance_from_query)

# Plot statistics
if PLOT_FLAG:
    plt.figure(figsize=(7, 4.5))
    plt.plot(num_candidates_history, linewidth=4)
    plt.xlabel('Search radius')
    plt.ylabel('# of documents searched')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    plt.figure(figsize=(7, 4.5))
    plt.plot(query_time_history, linewidth=4)
    plt.xlabel('Search radius')
    plt.ylabel('Query time (seconds)')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    plt.figure(figsize=(7, 4.5))
    plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
    plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
    plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
    plt.xlabel('Search radius')
    plt.ylabel('Cosine distance of neighbors')
    plt.legend(loc='best', prop={'size': 15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# Randomly choose 10 documents to run queries.
# First compute the true 25 nearest neighbors, and then run LSH multiple times.
# look at two metrics:
# 1. Precision@10: the number the 10 neighbors given by LSH are among the true 25 nearest neighbors
# 2. Average cosine distance of the neighbors from the query
print("Randomly choose 10 documents to run queries:")
max_radius = 17
precision = {i: [] for i in range(max_radius)}
average_distance = {i: [] for i in range(max_radius)}
query_time = {i: [] for i in range(max_radius)}

np.random.seed(0)
num_queries = 10
for i, ix in enumerate(np.random.choice(corpus.shape[0], num_queries, replace=False)):
    print('%s / %s' % (i+1, num_queries))

    # Get the set of 25 true nearest neighbors
    ground_truth = set(brute_force_query(corpus[ix, :], corpus, k=25)['id'])

    for r in range(1, max_radius):
        start = time.time()
        result, num_candidates = query(corpus[ix, :], model, k=10, max_search_radius=r)
        end = time.time()

        query_time[r].append(end - start)
        # precision = (# of neighbors both in result and ground_truth) / 10.0
        precision[r].append(len(set(result['id']) & ground_truth) / 10.0)
        average_distance[r].append(result['distance'][1:].mean())
print()

# Plot
if PLOT_FLAG:
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, 17), [np.mean(average_distance[i]) for i in range(1, 17)], linewidth=4,
             label='Average over 10 neighbors')
    plt.xlabel('Search radius')
    plt.ylabel('Cosine distance')
    plt.legend(loc='best', prop={'size': 15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, 17), [np.mean(precision[i]) for i in range(1, 17)], linewidth=4, label='Precison@10')
    plt.xlabel('Search radius')
    plt.ylabel('Precision')
    plt.legend(loc='best', prop={'size': 15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, 17), [np.mean(query_time[i]) for i in range(1, 17)], linewidth=4, label='Query time')
    plt.xlabel('Search radius')
    plt.ylabel('Query time (seconds)')
    plt.legend(loc='best', prop={'size': 15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# Change the number of random vectors in LSH algorithm (number of bits)
print("Change the number of random vectors in LSH algorithm (number of bits):")
precision = {i: [] for i in range(5, 20)}
average_distance = {i: [] for i in range(5, 20)}
query_time = {i: [] for i in range(5, 20)}
num_candidates_history = {i: [] for i in range(5, 20)}
ground_truth = {}

np.random.seed(0)
num_queries = 10
docs = np.random.choice(corpus.shape[0], num_queries, replace=False)

for i, ix in enumerate(docs):
    ground_truth[ix] = set(brute_force_query(corpus[ix, :], corpus, k=25)['id'])
    # Get the set of 25 true nearest neighbors

for num_vector in range(5, 20):
    print('num_vector = %s' % num_vector)
    model = train_lsh(corpus, num_vector, seed=143)

    for i, ix in enumerate(docs):
        start = time.time()
        result, num_candidates = query(corpus[ix, :], model, k=10, max_search_radius=3)
        end = time.time()

        query_time[num_vector].append(end - start)
        precision[num_vector].append(len(set(result['id']) & ground_truth[ix]) / 10.0)
        average_distance[num_vector].append(result['distance'][1:].mean())
        num_candidates_history[num_vector].append(num_candidates)
print()

# Plot
if PLOT_FLAG:
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(5, 20), [np.mean(average_distance[i]) for i in range(5, 20)], linewidth=4,
             label='Average over 10 neighbors')
    plt.xlabel('# of random vectors')
    plt.ylabel('Cosine distance')
    plt.legend(loc='best', prop={'size': 15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(5, 20), [np.mean(precision[i]) for i in range(5, 20)], linewidth=4, label='Precison@10')
    plt.xlabel('# of random vectors')
    plt.ylabel('Precision')
    plt.legend(loc='best', prop={'size': 15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(5, 20), [np.mean(query_time[i]) for i in range(5, 20)], linewidth=4, label='Query time (seconds)')
    plt.xlabel('# of random vectors')
    plt.ylabel('Query time (seconds)')
    plt.legend(loc='best', prop={'size': 15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(5, 20), [np.mean(num_candidates_history[i]) for i in range(5, 20)], linewidth=4,
             label='# of documents searched')
    plt.xlabel('# of random vectors')
    plt.ylabel('# of documents searched')
    plt.legend(loc='best', prop={'size': 15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


# QUIZ QUESTIONS:
print("Quiz Questions:")
# 1. What is the document id of Barack Obama's article?
print("1. The document ID of Barack Obama's article is {:d}.\n".format(obama_id))
# 2. Which bin contains Barack Obama's article? Enter its integer index.
print("2. The bin contains Barack Obama's article has index {:d}.\n".format(obama_bin))
# 3. Examine the bit representations of the bins containing Barack Obama and Joe Biden.
#    In how many places do they agree?
print("3. Biden's bin representation agrees with Obama's in {:d} out of {:d} places.\n"
      .format(np.sum(obama_bin_bits == biden_bin_bits), len(obama_bin_bits.tolist())))
# 4. What was the smallest search radius that yields the correct nearest neighbor, namely Joe Biden?
print("4. The smallest search radius that yields the correct nearest neighbor Joe Biden is {:d}.\n"
      .format(biden_min_radius))
# 5. Suppose our goal was to produce 10 approximate nearest neighbors
#    whose average distance from the query is within 0.01 of the average for the true 10 nearest neighbors.
#    For Barack Obama, the true 10 nearest neighbors are on average about 0.77.
#    What was the smallest search radius for Barack Obama that produced an average distance of 0.78 or better?
print("5. The smallest search radius that produces an average distance of <= 0.78 is {:d}."
      .format(np.min(np.argwhere(np.array(average_distance_from_query_history) <= 0.78))))

if PLOT_FLAG:
    plt.show()
