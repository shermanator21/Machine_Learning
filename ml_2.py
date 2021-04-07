
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3))

print(nyc.Date.values)

print(nyc.Date.values.reshape(-1, 1))

print(nyc.Temperature.values)

#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11)

print(X_train.shape)  # data
print(X_test.shape)  # data
print(y_train.shape)  # target
print(y_test.shape)  # target

#from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

# the fit method expects the samples ad the targets for training
linear_regression.fit(X=X_train, y=y_train)

print(linear_regression.coef_)

print(linear_regression.intercept_)


# trained, now to test the ml
predicted = linear_regression.predict(X_test)
expected = y_test

for p, e in zip(predicted[::5], expected[::5]):  # checks every 5th element
    print(f"predicted: {p:.2f}, expected: {e:.2f}")


# lambda impements y = mx + b


def predict(x): return linear_regression.coef_ * \
    x + linear_regression.intercept_


print(predict(2021))

print(predict(1890))

#import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False,
)

axes.set_ylim(10, 70)  # scale y-axis

#import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)
y = predict(x)
print(y)

#import matplotlib.pyplot as plt

line = plt.plot(x, y)
plt.show()
