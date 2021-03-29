from sklearn.datasets import load_digits

digits = load_digits()

print(digits.DESCR)  # contains the dataset's description

print(digits.data[13])  # numpy array that contain the 1719 sample

print(digits.data.shape)
