# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features.
# The dataset describes 50 samples for each of three Iris flower species: (Iris setosa, Iris versicolor, and Iris virginica)
# Each sample’s features are the sepal length, sepal width, petal length and petal width, all measured in centimeters.
# The sepals are the larger outer parts of each flower that protect the smaller inside petals before the flower buds bloom.

# EXERCISE
# 1. load the iris dataset and use classification to see if the expected and predicted species match up.

# 2. display the shape of the data, target, and target_names

# 3. display the first 10 predicted and expected results using the species names not the number (using target_names)

# 4. display the values that the model got wrong

# 5. visualize the data using the confusion matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 1.
iris = load_iris()
# print(iris.DESCR)

# 2.
print("Shape: ", iris.data.shape)
print("Target: ", iris['target'].shape)
print("Target Names: ", iris.target_names.shape)

# 3.
data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, random_state=11
)  # random_state for reproducibility

knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)  # uppercase X lowercase y
# Returns an array containing the predicted class of each test image:
# creates an array of digits

predicted = knn.predict(X=data_test)

expected = target_test

predicted_names = [(iris.target_names[p]) for p in predicted]
expected_names = [(iris.target_names[e]) for e in expected]

print("Predicted: ")

for i in range(0, 9):
    print(predicted_names[i])

for i in range(0, 9):
    print(expected_names[i])
print("Expected: ")

# 4
wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
print("Wrong: ", wrong)

# 5
confusion = confusion_matrix(y_true=expected, y_pred=predicted)

confusion_df = pd.DataFrame(confusion, index=range(3), columns=range(3))

figure = plt2.figure(figsize=(7, 6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)
# plt2.show()

print("done")
