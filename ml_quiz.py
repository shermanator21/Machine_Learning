from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import csv

# First read in the classes file and assign to panda database
classes = pd.read_csv("animal_classes.csv")

# create a dictionary with class number as key and class type as value
classDict = {}
n = 0

for i in classes.Class_Number:
    classDict[n + 1] = classes.Class_Type[n]
    n += 1

# TRAIN THE MODEL

# read in csv files to their own database with pandas
train_df = pd.read_csv("animals_train.csv")
test_df = pd.read_csv("animals_test.csv")

x_columns = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone',
             'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']
y_column = ["class_number"]

knn = KNeighborsClassifier()

knn.fit(X=train_df[x_columns], y=train_df[y_column])

# WRITING TO CSV
animal_predicted_classes = knn.predict(test_df[x_columns])

finalDict = {'animal_name': 'prediction'}
animalNames = test_df['animal_name']
i = 0

for number in animal_predicted_classes:
    finalDict[animalNames[i]] = classDict[number]
    i += 1

with open('predictions.csv', 'w') as f:
    for key in finalDict.keys():
        f.write("%s,%s\n" % (key, finalDict[key]))
