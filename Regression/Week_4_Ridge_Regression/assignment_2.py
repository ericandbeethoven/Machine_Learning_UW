import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PLOT_FLAG = False


def compute_RSS(y_predict, y_true):
    """
    Compute Residual Sum of Squares (RSS) of predicted data
    :param y_predict: Predicted Data
    :param y_true: True Data
    :return: float: RSS
    """
    return np.sum((y_predict - y_true) ** 2)


def get_numpy_data(data_sframe, features, output):
    """
    Get features (with padding) and output as numpy matrices
    :param data_sframe: input dataframe
    :param features: list of feature column names
    :param output: output column name
    :return:
    """
    return (np.hstack((np.ones((len(data_sframe), 1)), data_sframe[features].values)),
            data_sframe[output].values)


def predict_output(feature_matrix, weights):
    """
    Predict output of given features and weights
    :param feature_matrix: feature matrix with padding    n * d
    :param weights: vector of weights                     d
    :return: vector of predictions                        n
    """
    return np.dot(feature_matrix, weights)


def compute_cost(error, l2_penalty, weights):
    """
    Compute the cost of a model
    :param error: true output - predict output
    :param l2_penalty: lambda
    :param weights: weights of features
    :return: cost
    """
    return np.sum(error ** 2) + l2_penalty * np.sum(weights ** 2)


def compute_gradient(error, features, l2_penalty, weights):
    """
    Compute the gradient of weights
    :param error: true output - predict output
    :param features: features
    :param l2_penalty: lambda
    :param weights: weights of features
    :return: gradient
    """
    temp_weights = np.zeros(np.size(weights))
    temp_weights[1:] = weights[1:]
    return 2 * np.dot(error, features) + 2 * l2_penalty * temp_weights


def ridge_regression_gradient_descent(features, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    """
    Perform gradient descent on ridge regression
    :param features: features with padding 1
    :param output: output
    :param initial_weights: initial weights
    :param step_size: step size
    :param l2_penalty: lambda
    :param max_iterations: default set as =100
    :return:
    """
    weights = np.array(initial_weights)  # make sure it's a numpy array
    # while not reached maximum number of iterations:
    iter_count = 0
    while iter_count < max_iterations:
        # compute the predictions using your predict_output() function
        pred = predict_output(features, weights)
        # compute the errors as predictions - output
        err = pred - output

        grad = compute_gradient(err, features, l2_penalty, weights)
        weights -= step_size * grad
        iter_count += 1

    return weights


dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

house_data = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
house_test_data = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
house_train_data = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)

# Test
(example_features, example_output) = get_numpy_data(house_data, ['sqft_living'], 'price')
my_weights = np.array([10., 1.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output # prediction errors
print("Testing compute_gradient function. Values below should be the same.")
print(compute_gradient(errors, example_features, 1, my_weights))
print([np.sum(errors)*2., np.sum(errors*example_features[:, 1]) * 2 + 20.])
print()

# Part 1: Regression using feature 'sqft_living' and L2 penalty 0.0 and 1e11
print("Part 1: Train Rigid model using feature 'sqft_living' and L2 penalty 0.0 / 1e11")
feature_train_1, output_train_1 = get_numpy_data(house_train_data, ['sqft_living'], 'price')
feature_test_1, output_test_1 = get_numpy_data(house_test_data, ['sqft_living'], 'price')

step_size_1 = 1e-12
max_iter_1 = 1000
initial_weights_1 = np.zeros(2)

weights_1_no_penalty = ridge_regression_gradient_descent(feature_train_1, output_train_1, initial_weights_1,
                                                         step_size_1, 0, max_iter_1)

weights_1_large_penalty = ridge_regression_gradient_descent(feature_train_1, output_train_1, initial_weights_1,
                                                            step_size_1, 1e11, max_iter_1)


pred_train_1_initial = predict_output(feature_train_1, initial_weights_1)
pred_train_1_no_penalty = predict_output(feature_train_1, weights_1_no_penalty)
pred_train_1_large_penalty = predict_output(feature_train_1, weights_1_large_penalty)
pred_test_1_initial = predict_output(feature_test_1, initial_weights_1)
pred_test_1_no_penalty = predict_output(feature_test_1, weights_1_no_penalty)
pred_test_1_large_penalty = predict_output(feature_test_1, weights_1_large_penalty)

RSS_train_1_initial = compute_RSS(pred_train_1_initial, output_train_1)
RSS_train_1_no_penalty = compute_RSS(pred_train_1_no_penalty, output_train_1)
RSS_train_1_large_penalty = compute_RSS(pred_train_1_large_penalty, output_train_1)
RSS_test_1_initial = compute_RSS(pred_test_1_initial, output_test_1)
RSS_test_1_no_penalty = compute_RSS(pred_test_1_no_penalty, output_test_1)
RSS_test_1_large_penalty = compute_RSS(pred_test_1_large_penalty, output_test_1)

# Report
print("Model 1:")
print("Features: 'sqft_living'")
print("Output: 'price'")
print("Lambda: 0.0")
for i in range(2):
    print("\tw_{:s}: {:.6e}".format(str(i), weights_1_no_penalty[i]))
print()

print("Model 2:")
print("Features: 'sqft_living'")
print("Output: 'price'")
print("Lambda: 1e11")
for i in range(2):
    print("\tw_{:s}: {:.6e}".format(str(i), weights_1_large_penalty[i]))
print()

if PLOT_FLAG:
    plt.figure()
    plt.plot(feature_train_1[:, 1], output_train_1, '.',
             feature_train_1[:, 1], pred_train_1_no_penalty, '-',
             feature_train_1[:, 1], pred_train_1_large_penalty, '-')

    plt.figure()
    plt.plot(feature_test_1[:, 1], output_test_1, '.',
             feature_test_1[:, 1], pred_test_1_no_penalty, '-',
             feature_test_1[:, 1], pred_test_1_large_penalty, '-')

# Part 2: Multiple Regression using 'sqft_living' and 'sqft_living15'
print("Part 2: Multiple Regression using 'sqft_living' and 'sqft_living15'")
feature_train_2, output_train_2 = get_numpy_data(house_train_data, ['sqft_living', 'sqft_living15'], 'price')
feature_test_2, output_test_2 = get_numpy_data(house_test_data, ['sqft_living', 'sqft_living15'], 'price')

initial_weights_2 = np.array([0., 0., 0.])
step_size_2 = 1e-12
max_iter_2 = 1000

weights_2_no_penalty = ridge_regression_gradient_descent(feature_train_2, output_train_2, initial_weights_2,
                                                         step_size_2, 0, max_iter_2)

weights_2_large_penalty = ridge_regression_gradient_descent(feature_train_2, output_train_2, initial_weights_2,
                                                            step_size_2, 1e11, max_iter_2)

pred_train_2_initial = predict_output(feature_train_2, initial_weights_2)
pred_train_2_no_penalty = predict_output(feature_train_2, weights_2_no_penalty)
pred_train_2_large_penalty = predict_output(feature_train_2, weights_2_large_penalty)
pred_test_2_initial = predict_output(feature_test_2, initial_weights_2)
pred_test_2_no_penalty = predict_output(feature_test_2, weights_2_no_penalty)
pred_test_2_large_penalty = predict_output(feature_test_2, weights_2_large_penalty)

RSS_train_2_initial = compute_RSS(pred_train_2_initial, output_train_2)
RSS_train_2_no_penalty = compute_RSS(pred_train_2_no_penalty, output_train_2)
RSS_train_2_large_penalty = compute_RSS(pred_train_2_large_penalty, output_train_2)
RSS_test_2_initial = compute_RSS(pred_test_2_initial, output_test_2)
RSS_test_2_no_penalty = compute_RSS(pred_test_2_no_penalty, output_test_2)
RSS_test_2_large_penalty = compute_RSS(pred_test_2_large_penalty, output_test_2)

# Report
print("Model 1:")
print("Features: 'sqft_living', 'sqft_living15'")
print("Output: 'price'")
print("Lambda: 0.0")
for i in range(len(weights_2_no_penalty)):
    print("\tw_{:s}: {:.6e}".format(str(i), weights_2_no_penalty[i]))
print()

print("Model 2:")
print("Features: 'sqft_living', 'sqft_living15'")
print("Output: 'price'")
print("Lambda: 1e11")
for i in range(len(weights_2_large_penalty)):
    print("\tw_{:s}: {:.6e}".format(str(i), weights_2_large_penalty[i]))
print()


# QUIZ QUESTIONS:
print('Quiz Questions:\n')
# 1. In Part 1, what is the value of the coefficient for sqft_living that you learned
#    with no regularization? With high regularization?
print("1. In part 1, the value of the coefficient with no regulation is {:.1f}, with high regulation is {:.1f}."
      .format(weights_1_no_penalty[1], weights_1_large_penalty[1]))
print()

# 2. In Part 1, which line you fit with no regularization versus high regularization is steeper?
print("2. The line with no penalty is steeper.")
print()

# 3. In Part 1, what are the RSS on the test data for each of the set of weights above
#    (initial, no regularization, high regularization)?
print("3. In Part 1, the RSS on TEST data are:")
print("   RSS_initial : {:.8e}".format(RSS_test_1_initial))
print("   RSS_no_reg : {:.8e}".format(RSS_test_1_no_penalty))
print("   RSS_high_reg : {:.8e}".format(RSS_test_1_large_penalty))
print()

# 4. In Part 2, what is the value of the coefficient for sqft_living that you learned
#    with no regularization? With high regularization?
print("4. In part 2, the value of the coefficient with no regulation is {:.1f}, with high regulation is {:.1f}."
      .format(weights_2_no_penalty[1], weights_2_large_penalty[1]))
print()

# 5. In Part 2, what are the RSS on the test data for each of the set of weights above
#    (initial, no regularization, high regularization)?
print("5. In Part 2, the RSS on TEST data are:")
print("   RSS_initial : {:.8e}".format(RSS_test_2_initial))
print("   RSS_no_reg : {:.8e}".format(RSS_test_2_no_penalty))
print("   RSS_high_reg : {:.8e}".format(RSS_test_2_large_penalty))

# 6. In part 2, what's the error in predicting the price of the first house in the test set
#    using the weights learned with no regularization? What about with high regularization?
print("6. Price of the first house in the test set:  ${:.2f}".format(output_test_2[0]))
print("   Prediction with no regularization model:   ${:.2f} (error = {:.2f})"
      .format(pred_test_2_no_penalty[0], pred_test_2_no_penalty[0] - output_test_2[0]))
print("   Prediction with high regularization model: ${:.2f} (error = {:.2f})"
      .format(pred_test_2_large_penalty[0], pred_test_2_large_penalty[0] - output_test_2[0]))

if PLOT_FLAG:
    plt.show()