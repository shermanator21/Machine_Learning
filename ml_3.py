import matplotlib.pyplot as plt2
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()  # Bunch object

print(california.DESCR)

print(california.data.shape)

print(california.target.shape)

print(california.feature_names)


pd.set_option("precision", 4)
pd.set_option("max_columns", 9)
pd.set_option("display.width", None)

california_df = pd.DataFrame(california.data, columns=california.feature_names)

# add a column to the dataframe for the median house values store in califoria.target
california_df["MedHouseValue"] = pd.Series(california.target)

print(california_df.head())  # peek at first five rows

print(california_df.describe())

sample_df = california_df.sample(frac=0.1, random_state=17)


sns.set(font_scale=2)
sns.set_style("whitegrid")

for feature in california.feature_names:
    plt.figure(figsize=(8, 4.5))
    sns.scatterplot(
        data=sample_df,
        x=feature,
        y="MedHouseValue",
        hue="MedHouseValue",
        palette="cool",
        legend=False,
    )

# plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state=11)

print(X_train.shape)  # data
print(X_test.shape)  # data
print(y_train.shape)  # target
print(y_test.shape)  # target

linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)

print(linear_regression.coef_)
print(linear_regression.intercept_)

for i, name in enumerate(california.feature_names):
    print(f"{name}: {linear_regression.coef_[i]})")


predicted = linear_regression.predict(X_test)
print(predicted[:5])

expected = y_test
print(expected[:5])


# create a dataframe cotaining columns for the expected and predicted values
df = pd.DataFrame()

df["Expected"] = pd.Series(expected)
df["Predicted"] = pd.Series(predicted)

print(df[:10])


figure = plt2.figure(figsize=(9, 9))

axes = sns.scatterplot(
    data=df,
    x="Expected",
    y="Predicted",
    hue="Predicted",
    palette="cool",
    legend=False
)

start = min(expected.min(), predicted.min())
print(start)

end = max(expected.max(), predicted.max())
print(end)

axes.set_xlim(start, end)
axes.set_ylim(start, end)

line = plt2.plot([start, end], [start, end], "k--")

plt2.show()
