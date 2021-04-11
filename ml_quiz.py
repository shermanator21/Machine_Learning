from sklearn.linear_model import LinearRegression
import pandas as pd

# First read in the classes file and assign to panda database
classes = pd.read_csv("animal_classes.csv")
# print(classes)

# create a dictionary with class number as key and class type as value
classDict = {}
n = 0

for i in classes.Class_Number:
    classDict[n + 1] = classes.Class_Type[n]
    n += 1

# print(classDict)

# TRAIN THE MODEL
# read in training data
trainer = pd.read_csv("animals_train.csv")

X_train = trainer
y_train = trainer.class_number
# print(X_train)

linear_regression = LinearRegression()

# the fit method expects the samples and the targets for training
linear_regression.fit(X=X_train, y=y_train)
print(linear_regression.coef_)
print(linear_regression.intercept_)

# TEST THE MODEL
tester = pd.read_csv("animals_test.csv")
# print(tester)

predicted = linear_regression.predict(X_test)
print(predicted[:5])
