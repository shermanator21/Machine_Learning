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

train_df = pd.read_csv("animals_train.csv")


knn = KNeighborsClassifier()
# print(train_df['class_number'])


knn.fit(X=train_df, y=train_df['class_number'])

# TEST THE MODEL
test_df = pd.read_csv("animals_test.csv")

# convert all animal names to numbers
animalDict = {}
n = 0
for i in test_df.animal_name:
    animalDict[n + 1] = i
    n += 1

# replace animal names with numbers
'''
for n in range(1, 37):
    test_df["animal_name"].replace({animalDict[n]: n}, inplace=True)
'''

predicted = knn.predict(X=test_df)

expected = test_df['animal_name']

print(predicted[: 20])
print(expected[: 20])


# WRITE OUT TO CSV
'''
animals = pd.DateFrame([], columns=['animal_name', 'prediction'])

for animal in test:
    animals.append([animal, animal_prediction])

animals.to_csv('predictions.csv', index=False)
'''
