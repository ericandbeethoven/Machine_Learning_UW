import pandas as pd
import numpy as np
from math import sqrt

def compute_RSS(y_predict, y_true):
    """
    Compute Residual Sum of Squares (RSS) of predicted data
    :param y_predict: Predicted Data
    :param y_true: True Data
    :return: float: RSS
    """
    return np.sum((y_predict - y_true) ** 2)


def predict_outcome(feature_matrix, weights):
    """
    Predict output of given features and weights
    :param feature_matrix: feature matrix with padding    n * d
    :param weights: vector of weights                     d
    :return: vector of predictions                        n
    """
    return np.dot(feature_matrix, weights)


def feature_derivative(errors, feature):
    """
    The gradient function of regression cost function
    :param errors: Y_true - Y_predict             n
    :param feature: feature array                 n
    :return: gradient
    """
    return 2 * np.dot(errors, feature)


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    """
    Perform Gradient Descent
    :param feature_matrix: feature matrix with padding   n * d
    :param output: Y_true                                n
    :param initial_weights: initial weights              d
    :param step_size: step size
    :param tolerance: error tolerance to stop
    :return: result weights                              d
    """
    converged = False
    iterCount = 0
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        y_pred = predict_outcome(feature_matrix, weights)

        # compute the errors as predictions - output:
        y_err = y_pred - output
        gradient_sum_squares = 0  # initialize the gradient

        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            grad = feature_derivative(y_err, feature_matrix[:, i])

            # add the squared derivative to the gradient magnitude
            gradient_sum_squares += grad ** 2

            # update the weight based on step size and derivative:
            weights[i] -= step_size * grad

        gradient_magnitude = sqrt(gradient_sum_squares)

        if gradient_magnitude < tolerance:
            converged = True
    return weights


# Read data from csv file
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

house_data = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
house_test_data = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
house_train_data = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)

# Model 1:
# features: ‘sqft_living’
# output: ‘price’
# initial weights: -47000, 1 (intercept, sqft_living respectively)
# step_size = 7e-12
# tolerance = 2.5e7

# Model 2:
# features = ‘sqft_living’, ‘sqft_living_15’
# output = ‘price’
# initial weights = [-100000, 1, 1]
# step size = 4e-12
# tolerance = 1e9

# Extract features and output
num_train = np.size(house_train_data, 0)
num_test = np.size(house_test_data, 0)
output_train = house_train_data['price']
output_test = house_test_data['price']
feature_train_1 = np.vstack((np.ones(num_train), house_train_data['sqft_living'])).T
feature_test_1 = np.vstack((np.ones(num_test), house_test_data['sqft_living'])).T
feature_train_2 = np.hstack((np.ones((num_train, 1)), house_train_data[['sqft_living', 'sqft_living15']]))
feature_test_2 = np.hstack((np.ones((num_test, 1)), house_test_data[['sqft_living', 'sqft_living15']]))

# Initialize regression constants
init_weights_1 = np.array([-47000., 1.])
init_weights_2 = np.array([-100000., 1., 1.])
step_size_1 = 7e-12
step_size_2 = 4e-12
tolerance_1 = 2.5e7
tolerance_2 = 1e9

# Run gradient descent
weights_1 = regression_gradient_descent(feature_train_1, output_train, init_weights_1, step_size_1, tolerance_1)
weights_2 = regression_gradient_descent(feature_train_2, output_train, init_weights_2, step_size_2, tolerance_2)

# Linear Regression Report
print("Model 1:")
print("Features: 'sqft_living'")
print("Output: 'price'")
print("Weights:")
for i in range(len(weights_1)):
    print("\tw_{:s}: {:.4f}".format(str(i), weights_1[i]))
print()

print("Model 2:")
print("Features: 'sqft_living', 'sqft_living15'")
print("Output: 'price'")
print("Weights:")
for i in range(len(weights_2)):
    print("\tw_{:s}: {:.4f}".format(str(i), weights_2[i]))
print()

# Make prediction on test set and compute RSS on Model 1
output_pred_test_1 = predict_outcome(feature_test_1, weights_1)
RSS_test_1 = compute_RSS(output_pred_test_1, output_test)
output_pred_test_2 = predict_outcome(feature_test_2, weights_2)
RSS_test_2 = compute_RSS(output_pred_test_2, output_test)

# QUIZ QUESTIONS:
print('Quiz Questions:')
# What is the value of the weight for sqft_living in Model 1?
print("1. The weight for 'sqft_living' in Model 1 is {:.6f}.".format(weights_1[1]))
# What is the predicted price for the 1st house in the Test data set for model 1?
print("2. The predicted price for the 1st house in the TEST data set for Model 1 is ${:d}."
      .format(int(output_pred_test_1[0])))
# What is the predicted price for the 1st house in the TEST data set for model 2?
print("3. The predicted price for the 1st house in the TEST data set for Model 2 is ${:d}."
      .format(int(output_pred_test_2[0])))
# Which estimate was closer to the true price for the 1st house on the TEST data set, model 1 or model 2?
print("4. The true price is ${:d}. Model {:d} is closer to the true price."
      .format(int(output_test[0]),
              np.argmin([output_pred_test_1[0], output_pred_test_2[0]] - output_test[0]) + 1))
# Which model (1 or 2) has lowest RSS on all of the TEST data?
print("5. RSS(Model 1) = {:.3e}. RSS(Model 2) = {:.3e}. "
      "Model {:d} has lowest RSS on TEST data."
      .format(RSS_test_1, RSS_test_2, np.argmin([RSS_test_1, RSS_test_2]) + 1))