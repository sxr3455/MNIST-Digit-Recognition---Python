# Srinath Ramchandran, MNIST dataset classification using multi-layer perceptron network
# Libraries used, scikit-learn, sklearn, scipy, numpy

# Including necessary imports
import os.path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


print('Downloading MNIST dataset...')
mnist = fetch_mldata('MNIST original')

X, y = mnist.data, mnist.target


X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25) # defines split between trainset and testset


print('Got MNIST dataset with %d train and %d test samples!' % (len(y_train), len(y_test)))
print('Digit distribution in whole dataset:', np.bincount(y.astype('int64'))) # this prints the number of data points in each class



# Training
mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 10), learning_rate='adaptive', learning_rate_init=0.001, verbose=True)
mlp.fit(X_train, y_train)

#Testing

pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, pred) # accuracy calculation
print('Testing finished with an accuracy of :',accuracy*100,'%')



# Confusion Matrix
conf_matrix = confusion_matrix(y_test, pred)
print(conf_matrix)

