import pandas as pd
import numpy as np
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
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

house_data = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
house_test_data = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
house_train_data = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)

# Add new features
house_test_data['bedrooms_squared'] = house_test_data.bedrooms ** 2
house_test_data['bed_bath_rooms'] = house_test_data.bedrooms * house_test_data.bathrooms
house_test_data['log_sqft_living'] = np.log(house_test_data.sqft_living)
house_test_data['lat_plus_long'] = house_test_data.lat + house_test_data.long

house_train_data['bedrooms_squared'] = house_train_data.bedrooms ** 2
house_train_data['bed_bath_rooms'] = house_train_data.bedrooms * house_train_data.bathrooms
house_train_data['log_sqft_living'] = np.log(house_train_data.sqft_living)
house_train_data['lat_plus_long'] = house_train_data.lat + house_train_data.long

# Create linear models
output_train = house_train_data['price']
output_test = house_test_data['price']

# Model 1: 'sqft_living', 'bedrooms', 'bathrooms', 'lat' and 'long'
feature_train_1 = house_train_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']]
feature_test_1 = house_test_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']]
linear_model_1 = sklearn.linear_model.LinearRegression()
linear_model_1.fit(feature_train_1, output_train)
RSS_train_1 = compute_RSS(linear_model_1.predict(feature_train_1), output_train)
RSS_test_1 = compute_RSS((linear_model_1.predict(feature_test_1)), output_test)

print("Model 1:")
print("Features: 'sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long'")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.4f}".format(linear_model_1.intercept_))
for i in range(len(linear_model_1.coef_)):
    print("\tw_{:s}: {:.4f}".format(str(i+1), linear_model_1.coef_[i]))
print()

# Model 2: 'sqft_living', 'bedrooms', 'bathrooms', 'lat','long', and 'bed_bath_rooms'
feature_train_2 = house_train_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']]
feature_test_2 = house_test_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']]
linear_model_2 = sklearn.linear_model.LinearRegression()
linear_model_2.fit(feature_train_2, output_train)
RSS_train_2 = compute_RSS(linear_model_2.predict(feature_train_2), output_train)
RSS_test_2 = compute_RSS((linear_model_2.predict(feature_test_2)), output_test)

print("Model 2:")
print("Features: 'sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms'")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.4f}".format(linear_model_2.intercept_))
for i in range(len(linear_model_2.coef_)):
    print("\tw_{:s}: {:.4f}".format(str(i+1), linear_model_2.coef_[i]))
print()

# Model 3: 'sqft_living', 'bedrooms', 'bathrooms', 'lat','long',
#          'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living'
#          and 'lat_plus_long'
feature_train_3 = house_train_data[['sqft_living', 'bedrooms', 'bathrooms',
                                    'lat', 'long', 'bed_bath_rooms',
                                    'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']]
feature_test_3 = house_test_data[['sqft_living', 'bedrooms', 'bathrooms',
                                  'lat', 'long', 'bed_bath_rooms',
                                  'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']]
linear_model_3 = sklearn.linear_model.LinearRegression()
linear_model_3.fit(feature_train_3, output_train)
RSS_train_3 = compute_RSS(linear_model_3.predict(feature_train_3), output_train)
RSS_test_3 = compute_RSS((linear_model_3.predict(feature_test_3)), output_test)

print("Model 3:")
print("Features: 'sqft_living', 'bedrooms', 'bathrooms', 'lat','long', "
      "'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long'")
print("Output: 'price'")
print("Weights:\n\tw_0: {:.4f}".format(linear_model_3.intercept_))
for i in range(len(linear_model_3.coef_)):
    print("\tw_{:s}: {:.4f}".format(str(i+1), linear_model_3.coef_[i]))
print()

# Report RSS
print('{:15s}{:10s}{:10s}{:10s}'.format('', 'TRAINING', '','TESTING'))
print('{:15s}{:.5e}{:8s}{:.5e}'.format('Model 1', RSS_train_1, '', RSS_test_1))
print('{:15s}{:.5e}{:8s}{:.5e}'.format('Model 2', RSS_train_2, '', RSS_test_2))
print('{:15s}{:.5e}{:8s}{:.5e}'.format('Model 3', RSS_train_3, '', RSS_test_3))
print()

# QUIZ QUESTIONS:
print('Quiz Questions:')
# 1. What are the mean (arithmetic average) values of your 4 new variables on TEST data?
print('1. The mean values of new variables are:')
print('\tbedrooms_squared: {:.2f}'.format(np.mean(house_test_data.bedrooms_squared)))
print('\tbed_bath_rooms:   {:.2f}'.format(np.mean(house_test_data.bed_bath_rooms)))
print('\tlog_sqft_living:  {:.2f}'.format(np.mean(house_test_data.log_sqft_living)))
print('\tlat_plus_long:    {:.2f}'.format(np.mean(house_test_data.lat_plus_long)))

# 2. What is the sign for the weight for ‘bathrooms’ in Model 1?
print("2. The sign for the weight for 'bathrooms' in Model 1 is {:s}."
      .format('+' if linear_model_1.coef_[2] >= 0 else '-'))

# 3. What is the sign for the weight for ‘bathrooms’ in Model 2?
print("3. The sign for the weight for 'bathrooms' in Model 2 is {:s}."
      .format('+' if linear_model_2.coef_[2] >= 0 else '-'))

# 4. Which model (1, 2 or 3) had the lowest RSS on TRAINING data?
print("4. Model {:1d} has the lowest RSS on TRAINING data.".format(np.argmin([RSS_train_1, RSS_train_2, RSS_train_3]) + 1))

# 5. Which model (1, 2 or 3) had the lowest RSS on TESTING data?
print("5. Model {:1d} has the lowest RSS on TESTING data.".format(np.argmin([RSS_test_1, RSS_test_2, RSS_test_3]) + 1))