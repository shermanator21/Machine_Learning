import pandas as pd

classes = pd.read_csv("animal_classes.csv")

print(classes)

classDict = {}
i = 1

print(firstline)

for i in classes:
    classDict[i] = i[2]

print(classDict)
