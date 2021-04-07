''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

# 1. How many sameples and How many features?
print(diabetes.data.shape)
# 442 samples. 10 features

# 2. What does feature s6 represent?
print(diabetes.DESCR)
# glu (blood sugar level)

print(diabetes.data[13])
print(diabetes.target[13])

# 3. print out the coefficient
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11)

linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)

print(linear_regression.coef_)

# 4. print out the intercept
print(linear_regression.intercept_)

# 5. create a scatterplot with regression line

predicted = linear_regression.predict(X_test)
expected = y_test


plt.plot(expected, predicted, ".")

# plot a line, a perfit predict would all fall on this line
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)

plt.show()
