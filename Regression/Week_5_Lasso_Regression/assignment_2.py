import pandas as pd
import numpy as np
import math

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


def predict_output(feature_matrix, weights):
    """
    Predict output of given features and weights.
    :param feature_matrix: feature matrix with padding    n x d
    :param weights: vector of weights                     d
    :return: vector of predictions                        n
    """
    return np.dot(feature_matrix, weights)


def normalize_features(features):
    """
    Normalize all features (in column) by 2-norm.
    :param features: feature matrix
    :return: (normalized feature matrix, vector of 2-norm of each original feature)
    """
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms
    return normalized_features, norms


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    """
    Perform one step of coordinate descent algorithm to update weight along one coordinate.
    :param i: coordinate along feature i
    :param feature_matrix: feature matrix             n x d
    :param output: vector of real value of output     n
    :param weights: vector of weights                 d
    :param l1_penalty: L1 penalty
    :return: new weight of feature i
    """
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = np.dot(feature_matrix[:, i], (output - prediction + weights[i] * feature_matrix[:, i]))

    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2
    else:
        new_weight_i = 0.0

    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    """
    Perform cyclical coordinate descent to calculate LASSO regression.
    :param feature_matrix: feature matrix             n x d
    :param output: vector of real value of output     n
    :param initial_weights: vector of init weights    d
    :param l1_penalty: L1 penalty
    :param tolerance: terminate if the maximum change across all coordinates less than tolerance
    :return: vector of weights of all features        d
    """
    max_delta = tolerance + 1
    weights = np.copy(initial_weights)
    while max_delta > tolerance:
        max_delta = 0
        for i in range(len(weights)):
            new_weight_i = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            delta = np.abs(new_weight_i - weights[i])
            if delta > max_delta:
                max_delta = delta
            weights[i] = new_weight_i
    return weights


# Read data from csv file
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int,
              'sqft_living15': float, 'grade': int, 'yr_renovated': int,
              'price': float, 'bedrooms': float, 'zipcode': str,
              'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': float, 'condition': int, 'lat': float,
              'date': str, 'sqft_basement': int, 'yr_built': int,
              'id': str, 'sqft_lot': int, 'view': int}

house_data = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
house_test_data = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
house_train_data = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)

# Coordinate descent function test
print("Step Coordinate Descent test:")
test_result = lasso_coordinate_descent_step(1,
                                            np.array([[3. / math.sqrt(13), 1. / math.sqrt(10)],
                                                      [2. / math.sqrt(13), 3. / math.sqrt(10)]]),
                                            np.array([1., 1.]),
                                            np.array([1., 4.]),
                                            0.1)
print(test_result)
print(0.425558846691)
print()
np.testing.assert_almost_equal(test_result, 0.425558846691)

# Part 1: Simple model with 2 features, 'sqft_living' and 'bedrooms', on entire dataset
print("Part 1: Simple model with 2 features")
feature_1, output_1 = get_numpy_data(house_data, ['sqft_living', 'bedrooms'], 'price')
feature_normalized_1, feature_norm_1 = normalize_features(feature_1)
initial_weights_1 = [1., 4., 1.]
prediction_1 = predict_output(feature_normalized_1, initial_weights_1)
ro_1 = [0] * 3
for i in range(3):
    ro_1[i] = np.sum(feature_1[:, i] * (output_1 - prediction_1 + initial_weights_1[i] * feature_1[:, i]))
print("ro values of features: [{:6e}, {:6e}, {:6e}] \n".format(ro_1[0], ro_1[1], ro_1[2]))

# Part 2: Apply Cyclical Coordinate Descent on simple model with 2 features
print("Part 2: Apply Cyclical Coordinate Descent on simple model with 2 features")
feature_2, output_2 = get_numpy_data(house_data, ['sqft_living', 'bedrooms'], 'price')
feature_normalized_2, feature_norm_2 = normalize_features(feature_2)
initial_weights_2 = np.array([0., 0., 0.])
L1_penalty_2 = 1e7
tolerance_2 = 1.0
weights_2 = lasso_cyclical_coordinate_descent(feature_normalized_2, output_2, initial_weights_2,
                                              L1_penalty_2, tolerance_2)
RSS_2 = compute_RSS(predict_output(feature_normalized_2, weights_2), output_2)

# Report
print("Lambda: {:e}".format(L1_penalty_2))
print("Features: 'sqft_living', 'bedrooms'")
print("Weights:")
print(pd.Series(weights_2, index=['intercept', 'sqft_living', 'bedrooms']))
print("RSS: {:.10e}".format(RSS_2))
print()

# Part 3: LASSO Regression with more features
print("Part 3: LASSO Regression with more features")
feature_list_3 = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                  'floors', 'waterfront', 'view', 'condition', 'grade',
                  'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
feature_list_with_intercept_3 = ['intercept'] + feature_list_3

feature_train_3, output_train_3 = get_numpy_data(house_train_data, feature_list_3, 'price')
feature_test_3, output_test_3 = get_numpy_data(house_test_data, feature_list_3, 'price')

feature_train_normalized_3, feature_train_norm_3 = normalize_features(feature_train_3)
initial_weights_3 = np.zeros(len(feature_list_3) + 1)
tolerance_3_1 = 1.0
tolerance_3_2 = 1.0
tolerance_3_3 = 5e5
L1_penalty_3_1 = 1e7
L1_penalty_3_2 = 1e8
L1_penalty_3_3 = 1e4

weights_3_1 = lasso_cyclical_coordinate_descent(feature_train_normalized_3, output_train_3,
                                                initial_weights_3, L1_penalty_3_1, tolerance_3_1)

weights_3_2 = lasso_cyclical_coordinate_descent(feature_train_normalized_3, output_train_3,
                                                initial_weights_3, L1_penalty_3_2, tolerance_3_2)

weights_3_3 = lasso_cyclical_coordinate_descent(feature_train_normalized_3, output_train_3,
                                                initial_weights_3, L1_penalty_3_3, tolerance_3_3)

# Report
print("Features:" + str(feature_list_3))
print("Output: 'price'\n")

print("Model 1:")
print("Lambda: {:.1e}".format(L1_penalty_3_1))
print("Tolerance: {:.1e}".format(tolerance_3_1))
print("Weights:")
print(pd.Series(weights_3_1, index=feature_list_with_intercept_3))
print()

print("Model 2:")
print("Lambda: {:.1e}".format(L1_penalty_3_2))
print("Tolerance: {:.1e}".format(tolerance_3_2))
print("Weights:")
print(pd.Series(weights_3_2, index=feature_list_with_intercept_3))
print()

print("Model 3:")
print("Lambda: {:.1e}".format(L1_penalty_3_3))
print("Tolerance: {:.1e}".format(tolerance_3_3))
print("Weights:")
print(pd.Series(weights_3_3, index=feature_list_with_intercept_3))
print()

# Rescale weights
weights_normalized_3_1 = weights_3_1 / feature_train_norm_3
weights_normalized_3_2 = weights_3_2 / feature_train_norm_3
weights_normalized_3_3 = weights_3_3 / feature_train_norm_3

# Evaluating each of the learned models on the test data
RSS_test_3_1 = compute_RSS(predict_output(feature_test_3, weights_normalized_3_1), output_test_3)
RSS_test_3_2 = compute_RSS(predict_output(feature_test_3, weights_normalized_3_2), output_test_3)
RSS_test_3_3 = compute_RSS(predict_output(feature_test_3, weights_normalized_3_3), output_test_3)
print('RSS on TEST set:')
print("{:12s}{:10s}{:s}".format("L1_penalty", "", "RSS"))
print("{:.0e}{:12s}{:.6e}".format(L1_penalty_3_1, "", RSS_test_3_1))
print("{:.0e}{:12s}{:.6e}".format(L1_penalty_3_2, "", RSS_test_3_2))
print("{:.0e}{:12s}{:.6e}".format(L1_penalty_3_3, "", RSS_test_3_3))
print()

# QUIZ QUESTIONS:
print('Quiz Questions\n')
# 1. Recall that, whenever ro[i] falls between -l1_penalty/2 and l1_penalty/2,
#    the corresponding weight w[i] is sent to zero.
#    In part 1, now suppose we were to take one step of coordinate descent on either feature 1 or feature 2.
#    What range of values of l1_penalty would not set w[1] zero, but would set w[2] to zero,
#    if we were to take a step in that coordinate?
print("1. In part 1, we have ro_1={:.6e} and ro_2={:.6e}.".format(ro_1[1], ro_1[2]))
print("   To make w[1] not zero and w[2] zero, we have -l1_penalty/2 <= ro_2 <= l1_penalty/2 and ro_1 > l1_penalty/2.")
print("   Therefore, l1_penalty is in the range [{:.6e}, {:.6e}].".format(2 * ro_1[2], 2 * ro_1[1]))
print()

# 2. In part 1, what range of values of l1_penalty would set both w[1] and w[2] to zero?
print("2. In part 1, to make w[1] and w[2] both zero, we have -l1_penalty/2 <= ro_2 < ro_1 <= l1_penalty/2.")
print("   Since l1_penalty >= 0, left boundary is 0, we only need to satisfy ro_1 <= l1_penalty/2.")
print("   Therefore, l1_penalty is in the range [{:.6e}, Inf].\n".format(2 * ro_1[1]))

# 3. In part 2, what is the RSS of the learned model on the normalized dataset?
print("3. In part 2, RSS of the learned model on the normalized dataset is {:.10e}.\n".format(RSS_2))

# 4. In part 2, which features had weight zero at convergence?
zero_ind_2 = np.argwhere(weights_2 == 0)
feature_list_2 = ['intercept', 'sqft_living', 'bedrooms']
zero_feature_2 = [feature_list_2[i] for i in range(len(feature_list_2)) if i in zero_ind_2]
print("".join(["4. In part 2, '"] + zero_feature_2 + ["' has weight zero at convergence.\n"]))

# 5. In part 3, what features have non-zero weight in each case?
zero_ind_3_1 = np.argwhere(weights_3_1 == 0)
zero_ind_3_2 = np.argwhere(weights_3_2 == 0)
zero_ind_3_3 = np.argwhere(weights_3_3 == 0)
d_3 = len(feature_list_with_intercept_3)
nonzero_feature_3_1 = [feature_list_with_intercept_3[i] for i in range(d_3) if i not in zero_ind_3_1]
nonzero_feature_3_2 = [feature_list_with_intercept_3[i] for i in range(d_3) if i not in zero_ind_3_2]
nonzero_feature_3_3 = [feature_list_with_intercept_3[i] for i in range(d_3) if i not in zero_ind_3_3]
print("5. In Part 3 Model 1, " + str(nonzero_feature_3_1) + " have non-zero weight.")
print("   In Part 3 Model 2, " + str(nonzero_feature_3_2) + " have non-zero weight.")
print("   In Part 3 Model 3, " + str(nonzero_feature_3_3) + " have non-zero weight.\n")

# 6. In part 3, which model performed best on the test data?
best_L1_penalty = [L1_penalty_3_1, L1_penalty_3_2, L1_penalty_3_3][np.argmin([RSS_test_3_1, RSS_test_3_2, RSS_test_3_3])]
print("6. In part 3, the model with L1_penalty={:.0e} performs best on TEST data."
      .format(best_L1_penalty))