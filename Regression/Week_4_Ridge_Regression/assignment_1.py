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


def k_fold_cross_validation(k, l2_penalty, feature, output):
    """
    Perform k-fold cross validation on given lambda value
    :param k: number of folds
    :param l2_penalty: lambda value
    :param feature: input feature
    :param output: output
    :return: average_validation_error
    """
    n = len(feature)
    total_err = 0
    for i in range(k):
        start = n * i // k
        next_start = n * (i + 1) // k

        valid_feature = feature[start:next_start]
        train_feature = feature[0:start].append(feature[next_start:n])
        valid_output = output[start:next_start]
        train_output = output[0:start].append(output[next_start:n])

        ridge_model = sklearn.linear_model.Ridge(alpha=l2_penalty, normalize=True)
        ridge_model.fit(train_feature, train_output)
        total_err += compute_RSS(ridge_model.predict(valid_feature), valid_output)
    return total_err / k

# Read data from csv file
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

house_data = pd.read_csv('kc_house_data.csv', dtype=dtype_dict).sort_values(['sqft_living', 'price'])
house_test_data = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
house_train_data = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
house_valid_data = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)
house_set_1_data = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
house_set_2_data = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
house_set_3_data = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
house_set_4_data = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
house_trainvalid_data = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)

# Part 1: Use Rigid Regression
print("Part 1: Use Rigid Regression")

L2_penalty_1 = 1.5e-5
feature_1 = polynomial_dataframe(house_data['sqft_living'], 15)
output_1 = house_data['price']
ridge_model_1 = sklearn.linear_model.Ridge(alpha=L2_penalty_1, normalize=True)
ridge_model_1.fit(feature_1, output_1)

# Report
print("Model 1:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Lambda: {:e}".format(L2_penalty_1))
print("Weights:\n\tw_0: {:.6e}".format(ridge_model_1.intercept_))
for i in range(len(ridge_model_1.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), ridge_model_1.coef_[i]))
print()

# Part 2: Ridge Regression with small lambda
print("Part 2: Ridge Regression with small lambda")

L2_penalty_2 = 1e-9

feature_2_1 = polynomial_dataframe(house_set_1_data['sqft_living'], 15)
feature_2_2 = polynomial_dataframe(house_set_2_data['sqft_living'], 15)
feature_2_3 = polynomial_dataframe(house_set_3_data['sqft_living'], 15)
feature_2_4 = polynomial_dataframe(house_set_4_data['sqft_living'], 15)

output_2_1 = house_set_1_data['price']
output_2_2 = house_set_2_data['price']
output_2_3 = house_set_3_data['price']
output_2_4 = house_set_4_data['price']

ridge_model_2_1 = sklearn.linear_model.Ridge(alpha=L2_penalty_2, normalize=True)
ridge_model_2_2 = sklearn.linear_model.Ridge(alpha=L2_penalty_2, normalize=True)
ridge_model_2_3 = sklearn.linear_model.Ridge(alpha=L2_penalty_2, normalize=True)
ridge_model_2_4 = sklearn.linear_model.Ridge(alpha=L2_penalty_2, normalize=True)

ridge_model_2_1.fit(feature_2_1, output_2_1)
ridge_model_2_2.fit(feature_2_2, output_2_2)
ridge_model_2_3.fit(feature_2_3, output_2_3)
ridge_model_2_4.fit(feature_2_4, output_2_4)

# Report
print("Model 1:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Lambda: {:e}".format(L2_penalty_2))
print("Weights:\n\tw_0: {:.6e}".format(ridge_model_2_1.intercept_))
for i in range(len(ridge_model_2_1.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), ridge_model_2_1.coef_[i]))
print()

print("Model 2:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Lambda: {:e}".format(L2_penalty_2))
print("Weights:\n\tw_0: {:.6e}".format(ridge_model_2_2.intercept_))
for i in range(len(ridge_model_2_2.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), ridge_model_2_2.coef_[i]))
print()

print("Model 3:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Lambda: {:e}".format(L2_penalty_2))
print("Weights:\n\tw_0: {:.6e}".format(ridge_model_2_3.intercept_))
for i in range(len(ridge_model_2_3.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), ridge_model_2_3.coef_[i]))
print()

print("Model 4:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Lambda: {:e}".format(L2_penalty_2))
print("Weights:\n\tw_0: {:.6e}".format(ridge_model_2_4.intercept_))
for i in range(len(ridge_model_2_4.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), ridge_model_2_4.coef_[i]))
print()

# Plot
if PLOT_FLAG:
    plt.figure()
    plt.plot(feature_2_1['power_1'], output_2_1, '.',
             feature_2_1['power_1'], ridge_model_2_1.predict(feature_2_1), '-')
    plt.figure()
    plt.plot(feature_2_2['power_1'], output_2_2, '.',
             feature_2_2['power_1'], ridge_model_2_2.predict(feature_2_2), '-')
    plt.figure()
    plt.plot(feature_2_3['power_1'], output_2_3, '.',
             feature_2_3['power_1'], ridge_model_2_3.predict(feature_2_3), '-')
    plt.figure()
    plt.plot(feature_2_4['power_1'], output_2_4, '.',
             feature_2_4['power_1'], ridge_model_2_4.predict(feature_2_4), '-')

# Part 3: Ridge Regression with large lambda
print("Part 3: Ridge Regression with small lambda")
L2_penalty_3 = 1.23e2

feature_3_1 = polynomial_dataframe(house_set_1_data['sqft_living'], 15)
feature_3_2 = polynomial_dataframe(house_set_2_data['sqft_living'], 15)
feature_3_3 = polynomial_dataframe(house_set_3_data['sqft_living'], 15)
feature_3_4 = polynomial_dataframe(house_set_4_data['sqft_living'], 15)

output_3_1 = house_set_1_data['price']
output_3_2 = house_set_2_data['price']
output_3_3 = house_set_3_data['price']
output_3_4 = house_set_4_data['price']

ridge_model_3_1 = sklearn.linear_model.Ridge(alpha=L2_penalty_3, normalize=True)
ridge_model_3_2 = sklearn.linear_model.Ridge(alpha=L2_penalty_3, normalize=True)
ridge_model_3_3 = sklearn.linear_model.Ridge(alpha=L2_penalty_3, normalize=True)
ridge_model_3_4 = sklearn.linear_model.Ridge(alpha=L2_penalty_3, normalize=True)

ridge_model_3_1.fit(feature_3_1, output_3_1)
ridge_model_3_2.fit(feature_3_2, output_3_2)
ridge_model_3_3.fit(feature_3_3, output_3_3)
ridge_model_3_4.fit(feature_3_4, output_3_4)

# Report
print("Model 1:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Lambda: {:e}".format(L2_penalty_3))
print("Weights:\n\tw_0: {:.6e}".format(ridge_model_3_1.intercept_))
for i in range(len(ridge_model_3_1.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), ridge_model_3_1.coef_[i]))
print()

print("Model 2:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Lambda: {:e}".format(L2_penalty_3))
print("Weights:\n\tw_0: {:.6e}".format(ridge_model_3_2.intercept_))
for i in range(len(ridge_model_3_2.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), ridge_model_3_2.coef_[i]))
print()

print("Model 3:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Lambda: {:e}".format(L2_penalty_3))
print("Weights:\n\tw_0: {:.6e}".format(ridge_model_3_3.intercept_))
for i in range(len(ridge_model_3_3.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), ridge_model_3_3.coef_[i]))
print()

print("Model 4:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Lambda: {:e}".format(L2_penalty_3))
print("Weights:\n\tw_0: {:.6e}".format(ridge_model_3_4.intercept_))
for i in range(len(ridge_model_3_4.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), ridge_model_3_4.coef_[i]))
print()

# Plot
if PLOT_FLAG:
    plt.figure()
    plt.plot(feature_3_1['power_1'], output_3_1, '.',
             feature_3_1['power_1'], ridge_model_3_1.predict(feature_3_1), '-')
    plt.figure()
    plt.plot(feature_3_2['power_1'], output_3_2, '.',
             feature_3_2['power_1'], ridge_model_3_2.predict(feature_3_2), '-')
    plt.figure()
    plt.plot(feature_3_3['power_1'], output_3_3, '.',
             feature_3_3['power_1'], ridge_model_3_3.predict(feature_3_3), '-')
    plt.figure()
    plt.plot(feature_3_4['power_1'], output_3_4, '.',
             feature_3_4['power_1'], ridge_model_3_4.predict(feature_3_4), '-')

# Part 4: Choosing Lambda via cross-validation
print("Part 4: Choosing Lambda via cross-validation")
feature_4 = polynomial_dataframe(house_trainvalid_data['sqft_living'], 15)
output_4 = house_trainvalid_data['price']
lambda_list = [10 ** (0.5 * x) for x in range(-10, 19)]
err_list = []
for i in range(len(lambda_list)):
    err_list.append(k_fold_cross_validation(10, lambda_list[i], feature_4, output_4))
    print("Lambda: {:.5e}, avg_RSS: {:.8e}".format(lambda_list[i], err_list[i]))

if PLOT_FLAG:
    # print RSS error on differe lambda
    plt.figure()
    plt.xscale('log')
    plt.plot(lambda_list, err_list, '-')

best_lambda = lambda_list[np.argmin(err_list)]
print("Chosen lambda is {:.4e}".format(best_lambda))
print()

# Train a new model on entire train_valid dataset
ridge_model_4 = sklearn.linear_model.Ridge(alpha=best_lambda, normalize=True)
ridge_model_4.fit(feature_4, output_4)

# Report
print("Model 1:")
print("Features: 'sqft_living' 15th order polynomial")
print("Output: 'price'")
print("Lambda: {:e}".format(best_lambda))
print("Weights:\n\tw_0: {:.6e}".format(ridge_model_4.intercept_))
for i in range(len(ridge_model_4.coef_)):
    print("\tw_{:s}: {:.6e}".format(str(i+1), ridge_model_4.coef_[i]))
print()

# Predict on test set
feature_test = polynomial_dataframe(house_test_data['sqft_living'], 15)
output_test = house_test_data['price']

output_test_pred_4 = ridge_model_4.predict(feature_test)
RSS_pred_4 = compute_RSS(output_test_pred_4, output_test)

if PLOT_FLAG:
    plt.figure()
    plt.plot(feature_test['power_1'], output_test, '.',
             feature_test['power_1'], output_test_pred_4, '.')


# QUIZ QUESTIONS:
print('Quiz Questions\n')
# 1. In Part 1, what’s the learned value for the coefficient of feature power_1?
print("1. In Part 1, the learned value for the coefficient of feature power_1 is {:.6e}.\n"
      .format(ridge_model_1.coef_[0]))

# 2. In Part 2, for the models learned in each of these training sets,
#    what are the smallest and largest values you learned for the coefficient of feature power_1?
power_1_coef_2 = [ridge_model_2_1.coef_[0], ridge_model_2_2.coef_[0],
                  ridge_model_2_3.coef_[0], ridge_model_2_4.coef_[0]]
print("2. In Part 2, the coefficients of feature power_1 are {:.4f}, {:.4f}, {:.4f}, {:.4f}."
      .format(*power_1_coef_2))
print("   The smallest value is {:.6e}.".format(min(power_1_coef_2)))
print("   The largest value is {:.6e}.".format(max(power_1_coef_2)))
print()

# 3. In Part 3, for the models learned in each of these training sets,
#    what are the smallest and largest values you learned for the coefficient of feature power_1?
power_1_coef_3 = [ridge_model_3_1.coef_[0], ridge_model_3_2.coef_[0],
                  ridge_model_3_3.coef_[0], ridge_model_3_4.coef_[0]]
print("3. In Part 3, the coefficients of feature power_1 are {:.4f}, {:.4f}, {:.4f}, {:.4f}."
      .format(*power_1_coef_3))
print("   The smallest value is {:.6e}.".format(min(power_1_coef_3)))
print("   The largest value is {:.6e}.".format(max(power_1_coef_3)))
print()

# 4. In Part 4, what is the best value for the L2 penalty according to 10-fold validation?
print("4. In Part 4, the best value for the L2 penalty is {:.4e}.\n"
      .format(best_lambda))

# 5. In Part 4, what is the RSS on the TEST data of the model you learn with this L2 penalty?
print("5. In Part 4, the RSS on the TEST data is {:.8e}.".format(RSS_pred_4))

# Make Plot
if PLOT_FLAG:
    plt.show()