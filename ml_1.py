import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()

print(digits.DESCR)  # contains the dataset's description

print(digits.data[13])  # numpy array that contain the 1797 samples

print(digits.data.shape)

print(digits.target[13])

print(digits.target.shape)

print(digits.images[13])

# import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
# python zip function bundles the 3 iterables and produces one iterable

for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    # displays multichannel (RGB) or single-channel ("greyscale")
    # image data
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])  # remove x-axis tick marks
    axes.set_yticks([])  # remove y-axis tick marks
    axes.set_title(target)  # the target value of the image
plt.tight_layout()
# plt.show()

#from sklean.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11
)  # random_state for reproducibility

print(data_train.shape)

print(target_train.shape)

print(data_test.shape)

#from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

# load the training data into the model using the fit method
# NoteL the KNeighborsClassifier fit method does not do calculations, it just loads the model

knn.fit(X=data_train, y=target_train)  # uppercase X lowercase y
# Returns an array containing the predicted class of each test image:
# creates an arrya of digits

predicted = knn.predict(X=data_test)

expected = target_test

print(predicted[:20])
print(expected[:20])

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
