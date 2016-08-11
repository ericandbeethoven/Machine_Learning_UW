import pandas as pd
import numpy as np
from math import sqrt
import sklearn.linear_model


def compute_RSS(y_predict, y_true):
    """
    Compute Residual Sum of Squares (RSS) of predicted data
    :param y_predict: Predicted Data
    :param y_true: True Data
    :return: float: RSS
    """
    return np.sum((y_predict - y_true) ** 2)


# Read data from csv file
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int,
              'sqft_living15': float, 'grade': int, 'yr_renovated': int,
              'price': float, 'bedrooms': float, 'zipcode': str,
              'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': float, 'condition': int, 'lat': float,
              'date': str, 'sqft_basement': int, 'yr_built': int,
              'id': str, 'sqft_lot': int, 'view': int}

house_data = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
house_test_data = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
house_train_data = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
house_valid_data = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)

# Create new features
for dataset in [house_data, house_test_data, house_train_data, house_valid_data]:
    dataset['sqft_living_sqrt'] = dataset['sqft_living'].apply(sqrt)
    dataset['sqft_lot_sqrt'] = dataset['sqft_lot'].apply(sqrt)
    dataset['bedrooms_square'] = dataset['bedrooms'] * dataset['bedrooms']
    dataset['floors_square'] = dataset['floors'] * dataset['floors']

# Create feature list
all_features = ['bedrooms', 'bedrooms_square', 'bathrooms',
                'sqft_living', 'sqft_living_sqrt', 'sqft_lot',
                'sqft_lot_sqrt', 'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

# Extract features and output
features_all = house_data[all_features]
features_test = house_test_data[all_features]
features_train = house_train_data[all_features]
features_valid = house_valid_data[all_features]

output_all = house_data['price']
output_test = house_test_data['price']
output_train = house_train_data['price']
output_valid = house_valid_data['price']

# Part 1: Using the entire house dataset, learn regression weights using an L1 penalty of 5e2.
print("Part 1: Learn LASSO regression using an L1 penalty of 5e2 on entire dataset.")
L1_penalty_1 = 5e2

lasso_model_1 = sklearn.linear_model.Lasso(alpha=L1_penalty_1, normalize=True)
lasso_model_1.fit(features_all, output_all)

# Report
chosen_feature_list_1 = [all_features[i] for i in range(len(all_features)) if lasso_model_1.coef_[i] != 0.0]
print("Model 1:")
print("Lambda: {:e}".format(L1_penalty_1))
print("Chosen features: " + str(chosen_feature_list_1))
print("Weights:\n\tw_0: {:.6e}".format(lasso_model_1.intercept_))
for i in range(len(lasso_model_1.coef_)):
    if lasso_model_1.coef_[i] != 0.0:
        print("\tw_{:s}: {:.10e}".format(str(i+1), lasso_model_1.coef_[i]))
print()

# Part 2: Explore multiple values of L1 penalty using a validation set
print("Part 2: Explore multiple values of L1 penalty using a validation set")

L1_penalty_2_list = np.logspace(1, 7, num=13)
lasso_model_2_list = []
RSS_valid_2_list = []

for i in range(len(L1_penalty_2_list)):
    lasso_model_2 = sklearn.linear_model.Lasso(alpha=L1_penalty_2_list[i], normalize=True)
    lasso_model_2.fit(features_train, output_train)
    lasso_model_2_list.append(lasso_model_2)

    pred_valid_2 = lasso_model_2_list[i].predict(features_valid)
    RSS_valid_2 = compute_RSS(pred_valid_2, output_valid)
    RSS_valid_2_list.append(RSS_valid_2)

# RSS Report on validation set
print("{:12s}{:10s}{:s}".format("L1_penalty", "", "RSS on validation set"))
for i in range(len(L1_penalty_2_list)):
    print("{:10.6e}{:10s}{:.8e}".format(L1_penalty_2_list[i], "", RSS_valid_2_list[i]))
print()

L1_penalty_best_ind_2 = np.argmin(RSS_valid_2_list)
L1_penalty_best_2 = L1_penalty_2_list[L1_penalty_best_ind_2]
print("Chosen L1 penalty is {:6e}\n".format(L1_penalty_best_2))

# Use best model to compute RSS on test set
lasso_model_best_2 = lasso_model_2_list[L1_penalty_best_ind_2]
pred_test_2 = lasso_model_best_2.predict(features_test)
RSS_test_2 = compute_RSS(pred_test_2, output_test)

# Report
chosen_feature_list_2 = [all_features[i] for i in range(len(all_features)) if lasso_model_best_2.coef_[i] != 0.0]
print("Model 1:")
print("Lambda: {:e}".format(L1_penalty_best_2))
print("Chosen features: " + str(chosen_feature_list_2))
print("RSS on TEST data: {:.8e}".format(RSS_test_2))
print("Weights:\n\tw_0: {:.6e}".format(lasso_model_best_2.intercept_))
for i in range(len(lasso_model_best_2.coef_)):
    if lasso_model_best_2.coef_[i] != 0.0:
        print("\tw_{:s}: {:.10e}".format(str(i+1), lasso_model_best_2.coef_[i]))
print()

# Part 3: Limit to 7 features
print("Part 3: Limit to 7 features")

max_nonzeros = 7

# Explore large range of L1_penalty values
L1_penalty_large_3_list = np.logspace(1, 4, num=20)
nonzero_count_3_list = []
for i in range(len(L1_penalty_large_3_list)):
    lasso_model_3 = sklearn.linear_model.Lasso(alpha=L1_penalty_large_3_list[i], normalize=True)
    lasso_model_3.fit(features_train, output_train)
    nonzero_count_3_list.append(np.count_nonzero(lasso_model_3.coef_) + np.count_nonzero(lasso_model_3.intercept_))

# Non zero report
print("{:12s}{:10s}{:s}".format("L1_penalty", "",
                                "# of non-zero features"))
for i in range(len(nonzero_count_3_list)):
    print("{:10.6e}{:10s}{:10d}".format(L1_penalty_large_3_list[i], "",
                                        nonzero_count_3_list[i]))
print()

# Find the two ends of our desired narrow range of L1_penalty
L1_penalty_min_idx = -1
L1_penalty_max_idx = -1
for i in range(len(nonzero_count_3_list)):
    if nonzero_count_3_list[i] > max_nonzeros >= nonzero_count_3_list[i + 1]:
        L1_penalty_min_idx = i
    if nonzero_count_3_list[i - 1] >= max_nonzeros > nonzero_count_3_list[i]:
        L1_penalty_max_idx = i

L1_penalty_min = L1_penalty_large_3_list[L1_penalty_min_idx]
L1_penalty_max = L1_penalty_large_3_list[L1_penalty_max_idx]

# Exploring narrower range of L1_penalty
L1_penalty_small_3_list = np.linspace(L1_penalty_min, L1_penalty_max, 20)
lasso_model_3_list = []
nonzero_count_small_3_list = []
RSS_valid_3_list = []

for i in range(len(L1_penalty_small_3_list)):
    lasso_model_3 = sklearn.linear_model.Lasso(alpha=L1_penalty_small_3_list[i], normalize=True)
    lasso_model_3.fit(features_train, output_train)
    lasso_model_3_list.append(lasso_model_3)
    nonzero_count_small_3_list.append(np.count_nonzero(lasso_model_3.coef_) + np.count_nonzero(lasso_model_3.intercept_))

    pred_valid_3 = lasso_model_3_list[i].predict(features_valid)
    RSS_valid_3 = compute_RSS(pred_valid_3, output_valid)
    RSS_valid_3_list.append(RSS_valid_3)

# RSS Report on validation set
print("{:12s}{:10s}{:20s}{:10s}{:s}".format("L1_penalty", "",
                                            "# of non-zero features", "",
                                            "RSS on validation set"))
for i in range(len(L1_penalty_small_3_list)):
    print("{:10.6e}{:10s}{:10d}{:23s}{:.8e}".format(L1_penalty_small_3_list[i], "",
                                                    nonzero_count_small_3_list[i], "",
                                                    RSS_valid_3_list[i]))
print()

L1_penalty_max_count_fit_ind_3 = np.argwhere(np.array(nonzero_count_small_3_list) == max_nonzeros).flatten().tolist()
L1_penalty_best_ind_3 = (np.argmin([RSS_valid_3_list[i] for i in L1_penalty_max_count_fit_ind_3])
                         + L1_penalty_max_count_fit_ind_3[0])
L1_penalty_best_3 = L1_penalty_small_3_list[L1_penalty_best_ind_3]
lasso_model_best_3 = lasso_model_3_list[L1_penalty_best_ind_3]
print("Chosen L1 penalty is {:6e}\n".format(L1_penalty_best_3))

# Report
chosen_feature_list_3 = [all_features[i] for i in range(len(all_features)) if lasso_model_best_3.coef_[i] != 0.0]
print("Model 1:")
print("Lambda: {:e}".format(L1_penalty_best_3))
print("Chosen features: " + str(chosen_feature_list_3))
print("Weights:\n\tw_0: {:.6e}".format(lasso_model_best_3.intercept_))
for i in range(len(lasso_model_best_3.coef_)):
    if lasso_model_best_3.coef_[i] != 0.0:
        print("\tw_{:s}: {:.10e}".format(str(i+1), lasso_model_best_3.coef_[i]))
print()

# QUIZ QUESTIONS:
print('Quiz Questions')
# 1. In part 1, which features have been chosen by LASSO?
print("1. In part 1, LASSO chooses " + str(chosen_feature_list_1) + " as non-zero features.")
# 2. In part 2, which was the best value for the l1_penalty?
print("2. In part 2, L1_penalty={:.4e} gives the lowest RSS on validation set."
      .format(L1_penalty_best_2))
# 3. In part 2, using the best L1 penalty, how many nonzero weights do you have?
print("3. In part 2, using the best L1 penalty, there are {:d} nonzero weights."
      .format(np.count_nonzero(lasso_model_best_2.coef_) + np.count_nonzero(lasso_model_best_2.intercept_)))
# 4. In part 3, What values did you find for l1_penalty_min and l1_penalty_max?
print("4. In part 3, L1_penalty_min = {:.4f}, L1_penalty_max = {:.4f}"
      .format(L1_penalty_min, L1_penalty_max))
# 5. In part 3, what value of l1_penalty has the lowest RSS on the VALIDATION set
#               and has sparsity equal to ‘max_nonzeros’?
print("5. In part 3, L1_penalty={:.4e} gives the lowest RSS on validation set, and has sparsity equalsto {:d}."
      .format(L1_penalty_best_3, max_nonzeros))
# 6. In part 3, what features in this model have non-zero coefficients?
print("6. In part 3, LASSO chooses " + str(chosen_feature_list_3) + " as non-zero features.")