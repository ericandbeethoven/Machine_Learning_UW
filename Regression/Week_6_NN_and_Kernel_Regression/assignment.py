import pandas as pd
import numpy as np


def compute_RSS(y_predict, y_true):
    """
    Compute Residual Sum of Squares (RSS) of predicted data.
    :param y_predict: Predicted Data
    :param y_true: True Data
    :return: float: RSS
    """
    return np.sum((y_predict - y_true) ** 2)


def get_numpy_data(data_sframe, features, output):
    """
    Get features (with padding) and output as numpy matrices.
    :param data_sframe: input dataframe
    :param features: list of feature column names
    :param output: output column name
    :return: (feature matrix, output vector)
    """
    return (np.hstack((np.ones((len(data_sframe), 1)), data_sframe[features].values)),
            data_sframe[output].values)


def normalize_features(features):
    """
    Normalize all features (in column) by 2-norm.
    :param features: feature matrix
    :return: (normalized feature matrix, vector of 2-norm of each original feature)
    """
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms
    return normalized_features, norms


def compute_distances(feature_instances, features_query):
    """
    Compute the Euclidean distance of query feature vector to every features in another feature matrix
    :param features_instances: feature matrix with n feature vectors of d dimensions    n x d
    :param features_query: query feature vector                                         d
    :return: distance vector                                                            n
    """
    return np.linalg.norm(features_query - feature_instances, axis=1)


def k_nearest_neighbors(k, feature_train, feature_query):
    """
    Return the indices of k-nearest neighbors of query features vector in features matrix
    :param k: number of nearest neighbors
    :param feature_train: training feature matrix
    :param features_query: input query feature vector
    :return: row indices of k-nearest neighbors
    """
    distance = compute_distances(feature_train, feature_query)
    return np.argsort(distance)[0:k]


def predict_output_of_query(k, feature_train, output_train, feature_query):
    """
    Run k-nearest neighbors algorithm on single query,
    and return the average price of k-nearest neighbors as predicted value.
    :param k: number of nearest neighbors
    :param feature_train: training feature matrix       n x d
    :param output_train: output value of training set   n
    :param feature_query: input query feature vector    d
    :return: predicted value
    """
    return np.mean(output_train[k_nearest_neighbors(k, feature_train, feature_query)])


def predict_output(k, feature_train, output_train, feature_test):
    """
    Run k-nearest neighbors algorithm on whole dataset,
    and return the average price of k-nearest neighbors as predicted value.
    :param k: number of nearest neighbors
    :param feature_train: training feature matrix       n x d
    :param output_train: output value of training set   n
    :param feature_test: input query feature vector     d
    :return: predicted value
    """

    return (pd.DataFrame(feature_test)
            .apply(lambda row: predict_output_of_query(k, feature_train, output_train, row.values), axis=1)
            .values)


# Datatype
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': float, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

# Load data
house_data = pd.read_csv('kc_house_data_small.csv', dtype=dtype_dict)
house_test_data = pd.read_csv('kc_house_data_small_test.csv', dtype=dtype_dict)
house_train_data = pd.read_csv('kc_house_data_small_train.csv', dtype=dtype_dict)
house_valid_data = pd.read_csv('kc_house_data_validation.csv', dtype=dtype_dict)

# Extract Features
feature_list = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
                'sqft_living15', 'sqft_lot15']

feature_train, output_train = get_numpy_data(house_train_data, feature_list, 'price')
feature_valid, output_valid = get_numpy_data(house_valid_data, feature_list, 'price')
feature_test, output_test = get_numpy_data(house_test_data, feature_list, 'price')

# Normalize Features
feature_train_normalized, feature_train_norm = normalize_features(feature_train)
feature_valid_normalized = feature_valid / feature_train_norm
feature_test_normalized = feature_test / feature_train_norm

# Compute the Euclidean distance from the query house (1st house of the test set)
# to first 10 houses in the training set
distance_1_top10 = compute_distances(feature_train_normalized[0:10], feature_test_normalized[0])
print("Euclidean distance from the query house (1st house of the test set) to first 10 houses in TRAINING set:")
for i in range(10):
    print("{:d} : {:.8e}".format(i, distance_1_top10[i]))
print()

# Take the query house to be third house of the test set (features_test[2]).
distance_2 = compute_distances(feature_train_normalized, feature_test_normalized[2])

# Make predictions for the first 10 houses in the test set, using k=10.
prediction_test_top10 = predict_output(10, feature_train_normalized, output_train, feature_test_normalized[0:10])
print("Predicted values for Top 10 houses in TEST dataset:")
print(prediction_test_top10)
print()

# Choosing the best value of k using a validation set
k_list = range(1, 16)
RSS_valid_list = []
for k in k_list:
    prediction_valid = predict_output(k, feature_train_normalized, output_train, feature_valid_normalized)
    RSS_valid_list.append(compute_RSS(prediction_valid, output_valid))
k_best = k_list[np.argmin(RSS_valid_list)]

# Report
print('RSS on VALIDATION set:')
print(" {:5s}{:6s}{:s}".format("k", "", "RSS"))
for i in range(len(k_list)):
    print("{:2d}{:10s}{:.8e}".format(k_list[i], "", RSS_valid_list[i]))
print("Chosen k is: {:d}".format(k_best))
print()

# Using best k to predict TEST dataset
prediction_test = predict_output(k_best, feature_train_normalized, output_train, feature_test_normalized)
RSS_test = compute_RSS(prediction_test, output_test)
print("RSS on TEST set using k={} is: {:.8e}\n".format(k_best, RSS_test))

# QUIZ QUESTIONS:
print('Quiz Questions\n')
# 1. What is the Euclidean distance between the query house (1st house of the test set)
#    and the 10th house of the training set?
print("1. The Euclidean distance between the 1st house of TEST set and the 10th house of TRAINING set is {:.6e}.\n"
      .format(distance_1_top10[-1]))
# 2. Among the first 10 training houses, which house is the closest to the query house?
print("2. Among the first 10 training houses, the house with index {:d} is the closest to the query house.\n"
      .format(np.argmin(distance_1_top10)))
# 3. Take the query house to be third house of the test set (features_test[2]).
#    What is the index of the house in the training set that is closest to this query house?
print("3. Take the query house to be third house of the test set (features_test[2]).\n"
      "   The index of the house in the TRAINING set that is closest to this query house is {:d}.\n"
      .format(np.argmin(distance_2)))
# 4. What is the predicted value of the query house based on 1-nearest neighbor regression?
print("4. The predicted value of the query house based on 1-nearest neighbor regression is ${:.2f}.\n"
      .format(output_train[np.argmin(distance_2)]))
# 5. What are the indices of the 4 training houses closest to the query house?
print("5. The indices of the 4 training houses closest to the query house are "
      + str(k_nearest_neighbors(4, feature_train_normalized, feature_test_normalized[2]))
      + '.\n')
# 6. Predict the value of the query house using k-nearest neighbors with k=4.
print("6. The predicted value of the query house using k=4 is ${:.2f}.\n"
      .format(predict_output_of_query(4, feature_train_normalized, output_train, feature_test_normalized[2])))
# 7. Make predictions for the first 10 houses in the test set, using k=10.
#    What is the index of the house in this query set that has the lowest predicted value?
#    What is the predicted value of this house?
print("7. Make predictions for the first 10 houses in the test set, using k=10.")
print("   The index of the house in this query set that has the lowest predicted value is {:d}."
      .format(np.argmin(prediction_test_top10)))
print("   The predicted value of this house is ${:.2f}.\n"
      .format(np.min(prediction_test_top10)))
# 8. What is the RSS on the TEST data using the value of k found above?
print("8. The RSS on the TEST data is {:.8e}.\n".format(RSS_test))