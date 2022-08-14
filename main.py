# Yasemin Direk
# CMPE 442 Assignment 2

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def sigmoid_derivative(z):
        return z * (1.0 - z)

# this function finds the accuracy and prints the confusion matrix.
def find_accuracy(predicted, target):
    correct_prediction = 0

    for i in range(len(predicted)):
        # print("Actual class:", target[i])
        # print("Predicted class:", predicted[i])
        if(predicted[i] == target[i]):
            correct_prediction = correct_prediction + 1

    accuracy = correct_prediction / predicted.size # predict.size = 25
    print("Number of correct predictions:", correct_prediction)
    print("Accuracy: ", accuracy)
    print("Confusion Matrix:")
    matrix = confusion_matrix(target, predicted)
    print(matrix)
    df_cm = pd.DataFrame(matrix, index=[i for i in [0, 1, 2]], columns=[i for i in [0, 1, 2]])
    plt.figure(figsize=(8, 5))
    plt.title("Confusion Matrix")
    seaborn.heatmap(df_cm, annot=True)

    plt.show()

class NeuralNetwork:
    def __init__(self, inSize, sl2, clsSize, lrt):
        # Constructor expects:
        # inSize- input size, number of features
        # sl2 - number of neurons in the hidden layer
        # clsSize - number of classes, equals number of neurons in output layer
        # lrt - learning rate

        self.iSz=inSize
        self.oSz=clsSize
        self.hSz=sl2
        # Initial assignment of weights
        np.random.seed(42)  ## assigning seed so it generates the same random number all the time. Just to fix the result.
        self.weights1 = (np.random.rand(self.hSz,self.iSz+1)-0.5)/np.sqrt(self.iSz)
        self.weights2 = (np.random.rand(self.oSz,self.hSz+1)-0.5)/np.sqrt(self.hSz)

        self.output=np.zeros(clsSize)
        # self.output = 0
        self.layer1=np.zeros(self.hSz)
        self.eta=lrt

        print("----------------------------------")
        print("inSize =", inSize)
        print("sl2 =", sl2)
        print("clsSize =", clsSize)
        print("LRT =", lrt)

    # this function send forward single sample
    def feedforward(self, x):
        x_inp = np.insert(x,0,1)
        XW1 = np.dot(self.weights1, x_inp) # input for hidden layer
        self.layer1 = sigmoid(XW1) #  sigmoid(x1.w1) = output from hidden layer

        self.layer1 = np.insert(self.layer1,0,1)
        XW2 = np.dot(self.weights2, self.layer1) # input for output layer
        self.output = sigmoid(XW2)  #  sigmoid(x2.w2) = output from output layer

    # this function backpropagates errors of single sample
    def backprop(self, x, trg):
        sigma_3 = (trg - self.output)  # outer layer error
        output_derivative = sigmoid_derivative(self.layer1) # output derivative to use in formula
        sigma_2 = np.multiply(np.dot(self.weights2.T, sigma_3), output_derivative) # calculating error of out hidden layer
        x_inp = np.insert(x,0,1)
        delta1 = np.multiply.outer(sigma_2[1:], x_inp) # weights1 update
        delta2 = np.multiply.outer(sigma_3, self.layer1) # weights2 update

        return delta1, delta2

    # This function is called for training the data
    def fit(self, X, y, iterNo):

        m=np.shape(X)[0]
        for i in range(iterNo):
            D1=np.zeros(np.shape(self.weights1))
            D2=np.zeros(np.shape(self.weights2))
            # new_error=0
            for j in range(m):
                self.feedforward(X[j])
                yt = np.zeros(self.oSz)
                yt[int(y[j])] = 1   # the output is converted to vector, so if class of a sample is 1, then yt=[0 1 0]
                [delta1,delta2]= self.backprop(X[j],yt)
                D1=D1+delta1
                D2=D2+delta2
            self.weights1= self.weights1+self.eta*(D1/m)  # weights1 are updated only ones after one epoch
            self.weights2=self.weights2+self.eta*(D2/m)   # weights2 are updated only ones after one epoch

        print("iterNo =",iterNo)

    # This function is called for prediction
    def predict(self,X):

        m=np.shape(X)[0]
        y_proba=np.zeros((m,3))
        y = np.zeros(m)
        for i in range(m):
            self.feedforward(X[i])
            y_proba[i,:]=self.output   # the outputs of the network are the probabilities
            y[i] = np.argmax(self.output) # here we convert the probabilities to classes
        return y, y_proba


iris = datasets.load_iris()

X = iris.data
y = iris.target

y = y.reshape(150,1)
dataset = np.concatenate((X,y),axis=1)
np.random.shuffle(dataset)   # shuffling the data

X = dataset[:,0:4]  # first 4 elements
y = dataset[:,4]  # last element

# 100 samples for training
train_x = X[0:100,:]
train_y = y[0:100]

#  25 sample for validation
validation_x = X[100:125,:]
validation_y = y[100:125]

#  25 samples for testing
test_x = X[125:150,:]
test_y = y[125:150]

# fixed parameters
inSize = 4
clsSize = 3

# For part a

# LRT = 0.1
nn1 = NeuralNetwork(inSize, 2, clsSize, 0.1)
nn1.fit(train_x, train_y, 1000)
[y_predict, y_proba] = nn1.predict(validation_x)
accr1 = find_accuracy(y_predict, validation_y)

# LRT = 0.2
nn2 = NeuralNetwork(inSize, 2, clsSize, 0.2)
nn2.fit(train_x, train_y, 1000)
[y_predict, y_proba] = nn2.predict(validation_x)
accr2 = find_accuracy(y_predict, validation_y)

# LRT = 0.3
nn3 = NeuralNetwork(inSize, 2, clsSize, 0.3)
nn3.fit(train_x, train_y, 1000)
[y_predict, y_proba] = nn3.predict(validation_x)
accr3 = find_accuracy(y_predict, validation_y)

print()
# For the best network that I choose
# testing the best network on test set also
print("Best network:")
[y_predict, y_proba] = nn2.predict(test_x)
find_accuracy(y_predict, test_y)

# For part b

# sl2 = 2
nn4 = NeuralNetwork(inSize, 2, clsSize, 0.2)
nn4.fit(train_x, train_y, 500)
[y_predict, y_proba] = nn4.predict(validation_x)
accr4 = find_accuracy(y_predict, validation_y)

# sl2 = 3
nn5 = NeuralNetwork(inSize, 3, clsSize, 0.2)
nn5.fit(train_x, train_y, 500)
[y_predict, y_proba] = nn5.predict(validation_x)
accr5 = find_accuracy(y_predict, validation_y)

# sl2 = 4
nn6 = NeuralNetwork(inSize, 4, clsSize, 0.2)
nn6.fit(train_x, train_y, 500)
[y_predict, y_proba] = nn6.predict(validation_x)
accr6 = find_accuracy(y_predict, validation_y)

# For part c
print()
print("Network with best hyperparameters:")
nn8 = NeuralNetwork(inSize, 3, clsSize, 0.2)
nn8.fit(train_x, train_y, 1000)
# on validation set
print("Test on validation set:")
[y_predict, y_proba] = nn8.predict(validation_x)
accr8 = find_accuracy(y_predict, validation_y)
# on test set
print("Test on test set:")
[y_predict, y_proba] = nn8.predict(test_x)
find_accuracy(y_predict, test_y)


