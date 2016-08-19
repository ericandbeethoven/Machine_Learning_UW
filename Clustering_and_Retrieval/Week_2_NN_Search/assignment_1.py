import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
import json
import sklearn.neighbors
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import operator
import itertools

PLOT_FLAG = False

print('''
There is a known bug in pandas 0.18.x that causes justification broken on header line
when printing DataFrame using df.to_string(index=False) method.
See https://github.com/pydata/pandas/issues/13032 for detail.

The header lines in this report might look broken in the report.
Will be fixed when new versions of Pandas releases.
''')


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

    return scipy.sparse.csr_matrix((data, indices, indptr), shape)


def unpack_dict(matrix, map_index_to_word):
    """
    Unpack index to word dictionary according to word count matrix.
    :param matrix: word count matrix
    :param map_index_to_word: index to word dictionary
    :return: list of word count
    """
    table = [x[0] for x in sorted(map_index_to_word.items(), key=operator.itemgetter(1))]
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr

    num_doc = matrix.shape[0]

    return [{k: v for k, v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i + 1]]],
                                  data[indptr[i]:indptr[i + 1]].tolist())}
            for i in range(num_doc)]


def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    :param name: name of person
    :return: table of most frequent words
    """
    row = wiki[wiki['name'] == name]
    word_count_table = pd.DataFrame(list(row['word_count'].iloc[0].items()), columns=['word', 'count'])
    return word_count_table.sort_values('count', ascending=False)


def top_words_tf_idf(name):
    """
    Get a table of the words with highest TF-IDF weight in the given person's wikipedia page.
    :param name: name of person
    :return: table of words with highest TF-IDF weight
    """
    row = wiki[wiki['name'] == name]
    word_count_table = pd.DataFrame(list(row['tf-idf'].iloc[0].items()), columns=['word', 'weight'])
    return word_count_table.sort_values('weight', ascending=False)


def has_top_words(word_count_vector, common_words):
    """
    Return if common words are included in the given word cound vector.
    :param word_count_vector: word count vector for one article
    :param common_words: common words to be tested
    :return: if common words included in word count vector
    """
    # extract the keys of word_count_vector and convert it to a set
    unique_words_set = set(word_count_vector.keys())
    common_words_set = set(common_words)
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return common_words_set.issubset(unique_words_set)

# Load Data
wiki = pd.read_csv("../Data/people_wiki.csv")
wiki['id'] = wiki.index
word_count = load_sparse_csr("../Data/people_wiki_word_count.npz")
tf_idf = load_sparse_csr("../Data/people_wiki_tf_idf.npz")
with open("../Data/people_wiki_map_index_to_word.json") as json_data:
    map_index_to_word = json.load(json_data)

# Part 1: Find 10 nearest neighbor of Obama's article
print("Part 1: Find 10 nearest neighbor of Obama's article")
# Locate Obama's article
obama_id = wiki[wiki['name'] == 'Barack Obama'].index.tolist()[0]

# Find nearest neighbors using word count vectors
model_1 = sklearn.neighbors.NearestNeighbors(metric='euclidean', algorithm='brute')
model_1.fit(word_count)
distances_1, indices_1 = model_1.kneighbors(word_count[obama_id], n_neighbors=10)
neighbors_1 = pd.DataFrame({'distance': distances_1.flatten(), 'id': indices_1.flatten()})

# Report
result_df_1 = wiki.merge(neighbors_1, on='id').sort_values('distance')
print(result_df_1[['id', 'name', 'distance']].to_string(index=False))
print()

# Part 2: Analysis top words of Barack Obama and Francisco Barrio
print("Part 2: Analysis top words of Barack Obama and Francisco Barrio")
# Add word count column to DataFrame
wiki['word_count'] = unpack_dict(word_count, map_index_to_word)

# Get top words of Barack Obama and Francisco Barrio
obama_words = top_words('Barack Obama')
barrio_words = top_words('Francisco Barrio')

# Report
print("Top 10 words of Barack Obama:")
print(obama_words.head(10).to_string(index=False))
print()
print("Top 10 words of Francisco Barrio:")
print(barrio_words.head(10).to_string(index=False))
print()

# Combine two DataFrames
combined_words = (obama_words
                  .merge(barrio_words, on='word')
                  .sort_values('count_x', ascending=False)
                  .rename(columns={'count_x': 'Obama', 'count_y': 'Barrio'}))

print("Combined Top Words:")
print(combined_words.head(10).to_string(index=False))
print()

# Take top 5 combined words as common words
common_words = combined_words.head(5)['word'].tolist()

# Apply has_top_words to generate new column in wiki DataFrame
wiki['has_top_words'] = wiki['word_count'].apply(lambda x: has_top_words(x, common_words))
has_top_words_count = wiki.groupby(by='has_top_words').size()

# Measure the pairwise distance between the Wikipedia pages of
# Barack Obama, George W. Bush, and Joe Biden.
people_list = ['Barack Obama', 'George W. Bush', 'Joe Biden']
people_word_count = scipy.sparse.vstack([word_count[wiki[wiki['name'] == s].index.tolist()[0]] for s in people_list])
people_distance = euclidean_distances(people_word_count, people_word_count)

people_distance_dict = {}
for pair in itertools.combinations(range(3), 2):
    people_distance_dict[(people_list[pair[0]], people_list[pair[1]])] = people_distance[pair[0], pair[1]]

sorted_people_distance = sorted(people_distance_dict.items(), key=operator.itemgetter(1))

print("Pairwise Euclidean Distance between " + ", ".join(people_list) + ":")
print("{:20s}{:5s}{:20s}{:5s}{:10s}".format("Person 1", "", "Person 2", "", "Distance"))
for distance in sorted_people_distance:
    print("{:20s}{:5s}{:20s}{:5s}{:.6f}".format(distance[0][0], "", distance[0][1], "", distance[1]))
print()

# Get words appear both in Barack Obama and George W. Bush. Find 10 words show up most often in Obama
print("Top 10 common words by Obama and Bush:")
bush_words = top_words('George W. Bush')
combined_words_2 = (obama_words
                    .merge(bush_words, on='word')
                    .sort_values('count_x', ascending=False)
                    .rename(columns={'count_x': 'Obama', 'count_y': 'Bush'}))
print(combined_words_2.head(10).to_string(index=False))
print()

# Part 3: Extract the TF-IDF vectors
print("Part 3: Extract the TF-IDF vectors")

# Add TF-IDF column into wiki DatraFrame
wiki['tf-idf'] = unpack_dict(tf_idf, map_index_to_word)

# Find Nearest Neighbor using TF-IDF vectors
model_tf_idf_1 = sklearn.neighbors.NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf_1.fit(tf_idf)
distances_tf_idf_1_1, indices_tf_idf_1_1 = model_tf_idf_1.kneighbors(tf_idf[obama_id], n_neighbors=10)
neighbors_tf_idf_1_1 = pd.DataFrame({'distance': distances_tf_idf_1_1.flatten(), 'id': indices_tf_idf_1_1.flatten()})

# Report
result_df_tf_idf_1 = wiki.merge(neighbors_tf_idf_1_1, on='id').sort_values('distance')
print(result_df_tf_idf_1[['id', 'name', 'distance']].to_string(index=False))
print()

obama_tf_idf = top_words_tf_idf('Barack Obama')
schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')

# Report
print("Top 10 words of Barack Obama:")
print(obama_tf_idf.head(10).to_string(index=False))
print()
print("Top 10 words of Phil Schiliro:")
print(schiliro_tf_idf.head(10).to_string(index=False))
print()

# Combine two DataFrames
combined_tf_idf = (obama_tf_idf
                   .merge(schiliro_tf_idf, on='word')
                   .sort_values('weight_x', ascending=False)
                   .rename(columns={'weight_x': 'Obama', 'weight_y': 'Schiliro'}))

print("Combined Top Words:")
print(combined_tf_idf.head(10).to_string(index=False))
print()

# Take top 5 combined words as common words
common_tf_idf_words = combined_tf_idf.head(5)['word'].tolist()

# Apply has_top_words to generate new column in wiki DataFrame
wiki['has_top_words_tf_idf'] = wiki['word_count'].apply(lambda x: has_top_words(x, common_tf_idf_words))
has_top_words_tf_idf_count = wiki.groupby(by='has_top_words_tf_idf').size()

# Compute the euclidean distance between TF-IDF features of Obama and Biden
people_tf_idf = scipy.sparse.vstack([tf_idf[wiki[wiki['name'] == 'Barack Obama'].index.tolist()[0]],
                                     tf_idf[wiki[wiki['name'] == 'Joe Biden'].index.tolist()[0]]])
people_distance_tf_idf = euclidean_distances(people_tf_idf, people_tf_idf)[0, 1]

# Compute length of all documents
wiki['length'] = wiki['text'].apply(lambda x: len(x.split(' ')))

# Compute 100 nearest neighbors (euclidean distance) and display their length
print("10 nearest neighbors (euclidean distance) and their length:")
distances_tf_idf_1_2, indices_tf_idf_1_2 = model_tf_idf_1.kneighbors(tf_idf[obama_id], n_neighbors=100)
neighbors_tf_idf_1_2 = pd.DataFrame({'distance': distances_tf_idf_1_2.flatten(), 'id': indices_tf_idf_1_2.flatten()})
result_df_tf_idf_1_2 = wiki.merge(neighbors_tf_idf_1_2, on='id').sort_values('distance')
print(result_df_tf_idf_1_2.head(10)[['id', 'name', 'length', 'distance']].to_string(index=False))

# Plot document lengths of 100 nearest neighbors of Obama
if PLOT_FLAG:
    plt.figure(figsize=(10.5, 4.5))
    plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
             label='Entire Wikipedia', zorder=3, alpha=0.8)
    plt.hist(result_df_tf_idf_1_2['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
             label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
    plt.axvline(x=wiki['length'][wiki[wiki['name'] == 'Barack Obama'].index.tolist()[0]], color='k', linestyle='--', linewidth=4,
                label='Length of Barack Obama', zorder=2)
    plt.axvline(x=wiki['length'][wiki[wiki['name'] == 'Joe Biden'].index.tolist()[0]], color='g', linestyle='--', linewidth=4,
                label='Length of Joe Biden', zorder=1)
    plt.axis([0, 1000, 0, 0.04])

    plt.legend(loc='best', prop={'size': 15})
    plt.title('Distribution of document length')
    plt.xlabel('# of words')
    plt.ylabel('Percentage')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# Train another model using cosine distance
model_tf_idf_2 = sklearn.neighbors.NearestNeighbors(metric='cosine', algorithm='brute')
model_tf_idf_2.fit(tf_idf)
distances_tf_idf_2, indices_tf_idf_2 = model_tf_idf_2.kneighbors(tf_idf[obama_id], n_neighbors=100)
neighbors_tf_idf_2 = pd.DataFrame({'distance': distances_tf_idf_2.flatten(), 'id': indices_tf_idf_2.flatten()})

# Report
print("10 nearest neighbors (cosine distance) and their length:")
result_df_tf_idf_2 = wiki.merge(neighbors_tf_idf_2, on='id').sort_values('distance')
print(result_df_tf_idf_2.head(10)[['id', 'name', 'length', 'distance']].to_string(index=False))
print()

# Plot
if PLOT_FLAG:
    plt.figure(figsize=(10.5, 4.5))
    plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
             label='Entire Wikipedia', zorder=3, alpha=0.8)
    plt.hist(result_df_tf_idf_1_2['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
             label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
    plt.hist(result_df_tf_idf_2['length'], 50, color='b', edgecolor='None', histtype='stepfilled', normed=True,
             label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)
    plt.axvline(x=wiki['length'][wiki[wiki['name'] == 'Barack Obama'].index.tolist()[0]], color='k', linestyle='--',
                linewidth=4,
                label='Length of Barack Obama', zorder=2)
    plt.axvline(x=wiki['length'][wiki[wiki['name'] == 'Joe Biden'].index.tolist()[0]], color='g', linestyle='--',
                linewidth=4,
                label='Length of Joe Biden', zorder=1)
    plt.axis([0, 1000, 0, 0.04])
    plt.legend(loc='best', prop={'size': 15})
    plt.title('Distribution of document length')
    plt.xlabel('# of words')
    plt.ylabel('Percentage')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# Part 4: Problem with cosine distances: Tweets vs. long articles
print("Part 4: Problem with cosine distances: Tweets vs. long articles")
print("Sample Tweet: Democratic governments control law in response to popular act.\n")

tweet = {'act': 3.4597778278724887,
         'control': 3.721765211295327,
         'democratic': 3.1026721743330414,
         'governments': 4.167571323949673,
         'in': 0.0009654063501214492,
         'law': 2.4538226269605703,
         'popular': 2.764478952022998,
         'response': 4.261461747058352,
         'to': 0.04694493768179923}

word_indices = [map_index_to_word[word] for word in tweet.keys()]

tweet_tf_idf = scipy.sparse.csr_matrix((list(tweet.values()), ([0] * len(word_indices), word_indices)),
                                       shape=(1, tf_idf.shape[1]))
obama_tf_idf = tf_idf[obama_id]
print("The cosine distance between Obama's article and the tweet is {:.6e}."
      .format(cosine_distances(obama_tf_idf, tweet_tf_idf)[0, 0]))
print('''
With cosine distances, the tweet is "nearer" to Barack Obama.
Ignoring article lengths completely resulted in nonsensical results.
In practice, it is common to enforce maximum or minimum document lengths.
''')

# QUIZ QUESTIONS:
print("Quiz Questions:")
# 1. Among the words that appear in both Barack Obama and Francisco Barrio,
#    take the 5 that appear most frequently in Obama.
#    How many of the articles in the Wikipedia dataset contain all of those 5 words?
print("1. Among the words that appear in both Barack Obama and Francisco Barrio, ")
print("   take the 5 that appear most frequently in Obama.")
print("   There are {:d} articles in the Wikipedia dataset contain all of those 5 words.\n"
      .format(has_top_words_count[True]))

# 2. Measure the pairwise distance between the Wikipedia pages of
#    Barack Obama, George W. Bush, and Joe Biden.
#    Which of the three pairs has the smallest distance?
print("2. {:s} and {:s} has the smallest distance, which is {:.6f}.\n"
      .format(sorted_people_distance[0][0][0], sorted_people_distance[0][0][1], sorted_people_distance[0][1]))

# 3. Collect all words that appear both in Barack Obama and George W. Bush pages.
#    Out of those words, find the 10 words that show up most often in Obama's page.
print("3. The 10 words that show up most often in Obama's page, and also appear in Bush's page are:")
print("   " + str(combined_words_2.head(10)['word'].tolist()) + "\n")

# 4. Among the words that appear in both Barack Obama and Phil Schiliro,
#    take the 5 that appear most frequently in Obama.
#    How many of the articles in the Wikipedia dataset contain all of those 5 words?
print("4. Among the words that appear in both Barack Obama and Phil Schiliro, ")
print("   take the 5 that appear most frequently in Obama.")
print("   There are {:d} articles in the Wikipedia dataset contain all of those 5 words.\n"
      .format(has_top_words_tf_idf_count[True]))

# 5. Compute the Euclidean distance between TF-IDF features of Obama and Biden.
print("5. The Euclidean distance between TF-IDF features of Obama and Biden is {:.6f}.\n"
      .format(people_distance_tf_idf))

if PLOT_FLAG:
    plt.show()