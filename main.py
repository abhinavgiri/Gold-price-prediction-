import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Data collection and processing
# Loading csv file into a pandas dataframe

gold_data = pd.read_csv('D:\Programing\pythonProject\gold price prediction\gld_price_data.csv')

print(gold_data.head(), gold_data.tail())

# find correlation

correlation = gold_data.corr()
# print(correlation)


x = gold_data.drop(['Date', 'GLD'], axis=1)
y = gold_data['GLD']

# splitting testing and training data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# model training: random forest algorithm

regressor = RandomForestRegressor(n_estimators=100)

# model training

regressor.fit(x_train, y_train)

# model evaluation: prediction on test data

test_data_prediction = regressor.predict(x_test)
print(test_data_prediction)

error_score = metrics.r2_score(y_test, test_data_prediction)
print(error_score)

# compare actual value and predict value

y_test = list(y_test)
plt.plot(y_test, color='blue', label='Actual value')
plt.plot(test_data_prediction, color='orange', label='Predicted value')
plt.title("Actual Price vs Predict Value")
plt.xlabel('Number value')
plt.ylabel("Gold price")
plt.legend()
plt.show()
