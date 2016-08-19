import pandas as pd
import numpy as np

def simple_linear_regression(input_feature, output, printInfo=True):
    """
    Perform 2D closed form linear regression, return intercept and slope
    :param input_feature: numpy array of float
    :param output: numpy array of float
    :return: (intercept, slope): (float, float)
    """
    N = np.size(input_feature)
    sum_y = np.sum(output)
    sum_x = np.sum(input_feature)
    sum_xy = np.sum(input_feature * output)
    sum_x2 = np.sum(input_feature ** 2)
    slope = (sum_xy - sum_x * sum_y / N) / (sum_x2 - sum_x * sum_x / N)
    intercept = sum_y / N - slope * sum_x / N
    if printInfo:
        print("Intercept: {:.4f}".format(intercept))
        print("Slope: {:.4f}\n".format(slope))
    return (intercept, slope)


def get_regression_predictions(input_feature, intercept, slope):
    """
    Return the prediction given by input features and weights.
    :param input_feature: numpy array of float
    :param intercept: float
    :param slope: float
    :return: prediction: numpy array of float
    """
    return input_feature * slope + intercept


def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    """
    Compute RSS of given input and given model.
    :param input_feature: numpy array of float
    :param output: numpy array of float
    :param intercept: float
    :param slope: float
    :return: rss: float
    """
    return np.sum((output - get_regression_predictions(input_feature, intercept, slope)) ** 2)


def inverse_regression_predictions(output, intercept, slope):
    """
    Predict features based on given output.
    :param output: numpy array of float
    :param intercept: float
    :param slope: float
    :return: predicted_features: numpy array of float
    """
    return (output - intercept) / slope

# Read data from csv file
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

house_data = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
house_test_data = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
house_train_data = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)

# Extract Feature and Output
input_train_feature_sqft = house_train_data['sqft_living']
input_train_feature_bdr = house_train_data['bedrooms']
output_train = house_train_data['price']

input_test_feature_sqft = house_test_data['sqft_living']
input_test_feature_bdr = house_test_data['bedrooms']
output_test = house_test_data['price']

# Perform Linear Regression
intercept_sqft, slope_sqft = simple_linear_regression(input_train_feature_sqft, output_train)
intercept_bdr, slope_bdr = simple_linear_regression(input_train_feature_bdr, output_train)

# QUIZ QUESTIONS:
print("Quiz Questions:")
# What is the predicted price for a house with 2650 sqft?
print("1. The predicted price for a house with 2650 sqft is ${:.3f}\n"
      .format(get_regression_predictions(2650, intercept_sqft, slope_sqft)))

# What is the RSS for the simple linear regression using squarefeet to predict prices on TRAINING data?
print("2. The RSS for the simple linear regression using squarefeet to predict prices on TRAINING data is {:.4e}\n"
      .format(get_residual_sum_of_squares(input_train_feature_sqft, output_train, intercept_sqft, slope_sqft)))

# What is the estimated square-feet for a house costing $800,000?
print("3. The estimated square-feet for a house costing $800000 is {:.3f}\n"
      .format(inverse_regression_predictions(800000, intercept_sqft, slope_sqft)))

# Which model (square feet or bedrooms) has lowest RSS on TEST data?
rss_test_sqft = get_residual_sum_of_squares(input_test_feature_sqft, output_test, intercept_sqft, slope_sqft)
rss_test_bdr = get_residual_sum_of_squares(input_test_feature_bdr, output_test, intercept_bdr, slope_bdr)
print("4. On test data:")
print("\t\tRSS of square feet: {:.8e}\n\t\tRSS of bedrooms: {:.8e}".format(rss_test_sqft, rss_test_bdr))
print("   {:s} has lower RSS on TEST data".format("Square feet" if rss_test_sqft < rss_test_bdr else "Bedrooms"))

