import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

nyc = pd.read_csv('ave_yearly_temp_nyc_1895-2017.csv')
nyc.columns = ['Date', 'Temperature', 'Anomaly']
nyc.Date = nyc.Date.floordiv(100)
print(nyc.head(3))


X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11)

linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)

predicted = linear_regression.predict(X_test)
expected = y_test

for p, e in zip(predicted[::5], expected[::5]):  # checks every 5th element
    print(f"predicted: {p:.2f}, expected: {e:.2f}")


def predict(x): return linear_regression.coef_ * \
    x + linear_regression.intercept_


print(predict(2021))
print(predict(1890))

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False,
)

axes.set_ylim(10, 70)


x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)
y = predict(x)
print(y)


line = plt.plot(x, y)
plt.show()

# How does it compare to January trends?
# The data is more accurate along the regression line. Also, the temperature is higher as expected
