import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

PLOT_FLAG = False

def polynomial_dataframe(feature, degree):
    """
    Generates an data frame with the first column equal to ‘feature’
    and the remaining columns equal to
    ‘feature’ to increasing integer powers up to ‘degree’.
    :param feature: an array of feature
    :param degree: integer > 1
    :return: dataframe with higher order features
    """
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree + 1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = feature ** power
    return poly_dataframe


def compute_RSS(y_predict, y_true):
    """
    Compute Residual Sum of Squares (RSS) of predicted data
    :param y_predict: Predicted Data
    :param y_true: True Data
    :return: float: RSS
    """
    return np.sum((y_predict - y_true) ** 2)

# Read data from csv file
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

house_data = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
house_test_data = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
house_train_data = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
house_valid_data = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)
house_set_1_data = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
house_set_2_data = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
house_set_3_data = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
house_set_4_data = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

# PART 1: Different Polynomial Orders
print("Part 1: Different Polynomial Orders")
# Sort house_data with 'sqft_living' and 'price'
house_data_sorted = house_data.sort_values(['sqft_living', 'price'])

# Model 1: Linear Regression with feature 'sqft_living' on full data set
# Model 2: 2nd degree polynomial feature
# Model 3: 3rd degree polynomial feature
# Model 4: 15th degree polynomial feature
feature_1_1 = polynomial_dataframe(house_data_sorted['sqft_living'], 1)
feature_1_2 = polynomial_dataframe(house_data_sorted['sqft_living'], 2)
feature_1_3 = polynomial_dataframe(house_data_sorted['sqft_living'], 3)
feature_1_4 = polynomial_dataframe(house_data_sorted['sqft_living'], 15)

output_1 = house_data_sorted['price']

linear_model_1_1 = sklearn.linear_model.LinearRegression()
linear_model_1_1.fit(feature_1_1, output_1)
RSS_1_1 = compute_RSS(linear_model_1_1.predict(feature_1_1), output_1)

linear_model_1_2 = sklearn.linear_model.LinearRegression()
linear_model_1_2.fit(feature_1_2, output_1)
RSS_1_2 = compute_RSS(linear_model_1_2.predict(feature_1_2), output_1)

linear_model_1_3 = sklearn.linear_model.LinearRegression()
linear_model_1_3.fit(feature_1_3, output_1)
RSS_1_3 = compute_RSS(linear_model_1_3.predict(feature_1_3), output_1)

linear_model_1_4 = sklearn.linear_model.LinearRegression()
linear_model_1_4.fit(feature_1_4, output_1)
RSS_1_4 = compute_RSS(linear_model_1_4.predict(feature_1_4), output_1)

# Report
print("Model 1:")
print("Features: 'sqft_living'")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.6e}".format(linear_model_1_1.intercept_))
for i in range(len(linear_model_1_1.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), linear_model_1_1.coef_[i]))
print()

print("Model 2:")
print("Features: 'sqft_living' 2nd order polynomial")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.6e}".format(linear_model_1_2.intercept_))
for i in range(len(linear_model_1_2.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), linear_model_1_2.coef_[i]))
print()

print("Model 3:")
print("Features: 'sqft_living' 3rd order polynomial")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.6e}".format(linear_model_1_3.intercept_))
for i in range(len(linear_model_1_3.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), linear_model_1_3.coef_[i]))
print()

print("Model 4:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.6e}".format(linear_model_1_4.intercept_))
for i in range(len(linear_model_1_4.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), linear_model_1_4.coef_[i]))
print()

# Produce a scatter plot of the training data and model
if PLOT_FLAG:
    plt.figure()
    plt.plot(feature_1_1['power_1'], output_1, '.',
             feature_1_1['power_1'], linear_model_1_1.predict(feature_1_1), '-')

    plt.figure()
    plt.plot(feature_1_2['power_1'], output_1, '.',
             feature_1_2['power_1'], linear_model_1_2.predict(feature_1_2), '-')

    plt.figure()
    plt.plot(feature_1_3['power_1'], output_1, '.',
             feature_1_3['power_1'], linear_model_1_3.predict(feature_1_3), '-')

    plt.figure()
    plt.plot(feature_1_4['power_1'], output_1, '.',
             feature_1_4['power_1'], linear_model_1_4.predict(feature_1_4), '-')

# RSS Report
print("RSS on whole data set:")
print("Model 1: {:.6e}".format(RSS_1_1))
print("Model 2: {:.6e}".format(RSS_1_2))
print("Model 3: {:.6e}".format(RSS_1_3))
print("Model 4: {:.6e}".format(RSS_1_4))
print()

# PART 2: Estimate a 15th degree polynomial on all 4 subsets
print("Part 2: Estimate a 15th degree polynomial on all 4 sets")

feature_2_1 = polynomial_dataframe(house_set_1_data['sqft_living'], 15)
feature_2_2 = polynomial_dataframe(house_set_2_data['sqft_living'], 15)
feature_2_3 = polynomial_dataframe(house_set_3_data['sqft_living'], 15)
feature_2_4 = polynomial_dataframe(house_set_4_data['sqft_living'], 15)
output_2_1 = house_set_1_data['price']
output_2_2 = house_set_2_data['price']
output_2_3 = house_set_3_data['price']
output_2_4 = house_set_4_data['price']

linear_model_2_1 = sklearn.linear_model.LinearRegression()
linear_model_2_1.fit(feature_2_1, output_2_1)
RSS_2_1 = compute_RSS(linear_model_2_1.predict(feature_2_1), output_2_1)

linear_model_2_2 = sklearn.linear_model.LinearRegression()
linear_model_2_2.fit(feature_2_2, output_2_2)
RSS_2_2 = compute_RSS(linear_model_2_2.predict(feature_2_2), output_2_2)

linear_model_2_3 = sklearn.linear_model.LinearRegression()
linear_model_2_3.fit(feature_2_3, output_2_3)
RSS_2_3 = compute_RSS(linear_model_2_3.predict(feature_2_3), output_2_3)

linear_model_2_4 = sklearn.linear_model.LinearRegression()
linear_model_2_4.fit(feature_2_4, output_2_4)
RSS_2_4 = compute_RSS(linear_model_2_4.predict(feature_2_4), output_2_4)

# Report
print("Model 1:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.6e}".format(linear_model_2_1.intercept_))
for i in range(len(linear_model_2_1.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), linear_model_2_1.coef_[i]))
print()

print("Model 2:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.6e}".format(linear_model_2_2.intercept_))
for i in range(len(linear_model_2_2.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), linear_model_2_2.coef_[i]))
print()

print("Model 3:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.6e}".format(linear_model_2_3.intercept_))
for i in range(len(linear_model_2_3.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), linear_model_2_3.coef_[i]))
print()

print("Model 4:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.6e}".format(linear_model_2_4.intercept_))
for i in range(len(linear_model_2_4.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), linear_model_2_4.coef_[i]))
print()

# Produce a scatter plot of the training data and model
if PLOT_FLAG:
    plt.figure()
    plt.plot(feature_2_1['power_1'], output_2_1, '.',
             feature_2_1['power_1'], linear_model_2_1.predict(feature_2_1), '-')
    plt.figure()
    plt.plot(feature_2_2['power_1'], output_2_2, '.',
             feature_2_2['power_1'], linear_model_2_2.predict(feature_2_2), '-')
    plt.figure()
    plt.plot(feature_2_3['power_1'], output_2_3, '.',
             feature_2_3['power_1'], linear_model_2_3.predict(feature_2_3), '-')
    plt.figure()
    plt.plot(feature_2_4['power_1'], output_2_4, '.',
             feature_2_4['power_1'], linear_model_2_4.predict(feature_2_4), '-')

# RSS Report
print("RSS on whole data set:")
print("Model 1: {:.6e}".format(RSS_2_1))
print("Model 2: {:.6e}".format(RSS_2_2))
print("Model 3: {:.6e}".format(RSS_2_3))
print("Model 4: {:.6e}".format(RSS_2_4))
print()

# Part 3: Cross Validation to select the best degree
print("Part 3: Cross Validation to select the best degree")
feature_train_raw = house_train_data['sqft_living']
feature_test_raw = house_test_data['sqft_living']
feature_valid_raw = house_valid_data['sqft_living']
output_train = house_train_data['price']
output_test = house_test_data['price']
output_valid = house_valid_data['price']

# Cross validation for degree from 1 to 15
RSS_valid = np.zeros(15)
for i in range(15):
    feature_train_temp = polynomial_dataframe(feature_train_raw, i + 1)
    feature_valid_temp = polynomial_dataframe(feature_valid_raw, i + 1)
    linear_model_3_temp = sklearn.linear_model.LinearRegression()
    linear_model_3_temp = linear_model_3_temp.fit(feature_train_temp, output_train)
    RSS_valid[i] = compute_RSS(linear_model_3_temp.predict(feature_valid_temp), output_valid)
    print('Model of polynomial degree {:d}, RSS = {:.8e}'.format(i + 1, RSS_valid[i]))

chosen_degree = np.argmin(RSS_valid) + 1
print('Chosen degree is {:d}'.format(chosen_degree))
print()

# Train final model on best degree
feature_train = polynomial_dataframe(feature_train_raw, chosen_degree)
feature_test = polynomial_dataframe(feature_test_raw, chosen_degree)
linear_model_3 = sklearn.linear_model.LinearRegression()
linear_model_3.fit(feature_train, output_train)
RSS_test = compute_RSS(linear_model_3.predict(feature_test), output_test)

print("Model 1:")
print("Features: 'sqft_living' {:d}-th order polynomial".format(chosen_degree))
print("Output: 'price'")
print("Weights:\n\tw_0: {:.6e}".format(linear_model_3.intercept_))
for i in range(len(linear_model_3.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), linear_model_3.coef_[i]))
print("\nRSS on TEST set:")
print("Model 1: {:.6e}\n".format(RSS_test))

# QUIZ QUESTIONS:
print('Quiz Questions:')
# 1. In part 2, is the sign (positive or negative) for power_15 the same in all four models?
sign_1 = np.sign(linear_model_2_1.coef_[-1])
sign_2 = np.sign(linear_model_2_2.coef_[-1])
sign_3 = np.sign(linear_model_2_3.coef_[-1])
sign_4 = np.sign(linear_model_2_4.coef_[-1])
print("1. In part 2, the signs for power_15 in all four models are:")
print("\tSet 1: {:s}".format('+' if sign_1 == 1 else '-'))
print("\tSet 2: {:s}".format('+' if sign_2 == 1 else '-'))
print("\tSet 3: {:s}".format('+' if sign_3 == 1 else '-'))
print("\tSet 4: {:s}".format('+' if sign_4 == 1 else '-'))

# 2. In part 2, True/False the plotted fitted lines look the same in all four plots?
print('2. In part 2 the plotted fitted lines does not look the same.')

# 3. In part 3, which degree (1, 2, …, 15) had the lowest RSS on Validation data?
print("3. Degree {:d} has the lowest RSS on validation data.".format(chosen_degree))

# 4. In part 3, what is the RSS on TEST data for the model with the degree selected from Validation data?
print("4. The RSS on TEST data of model with degree 6 is {:6e}".format(RSS_test))

# Show figures
if PLOT_FLAG:
    plt.show()
