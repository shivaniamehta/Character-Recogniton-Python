import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
"""
data = pd.read_csv('shuffle.csv', header=None)
#print(data.head())

y = data[0]
X = data.drop(data[0], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)
print(len(X_train))
print(len(X_test))

"""
from psutil import virtual_memory
import matplotlib.pyplot as plt

image_size = 28  # width and length
no_of_different_labels = 26  # i.e. 0, 1, 2, 3, ..., 25
image_pixels = image_size * image_size

train_data = np.empty([60001, 785])
row = 0
for line in open("train1.csv"):
    d = np.fromstring(line, sep=",")
    train_data[row] = np.fromstring(line, sep=",")
    row += 1
    if (row == 60001):
        break;

test_data = np.empty([10001, 785])
row = 0
for line in open("test1.csv"):
    d = np.fromstring(line, sep=",")
    test_data[row] = np.fromstring(line, sep=",")
    row += 1
    if row == 10001:
        break;

fac = 255 * 0.99 + 0.01

train_imgs = np.asfarray(train_data[:, 1:]) / fac
test_imgs = np.asfarray(test_data[:, 1:]) / fac
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

lr = np.arange(26)
for label in range(26):
    one_hot = (lr == label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", one_hot)

lr = np.arange(no_of_different_labels)
# transform labels into one hot representation
train_labels_one_hot = (lr == train_labels).astype(np.float)
test_labels_one_hot = (lr == test_labels).astype(np.float)
train_labels_one_hot[train_labels_one_hot == 0] = 0.01
train_labels_one_hot[train_labels_one_hot == 1] = 0.99
test_labels_one_hot[test_labels_one_hot == 0] = 0.01
test_labels_one_hot[test_labels_one_hot == 1] = 0.99

"""for i in range(10):
    img = train_imgs[i].reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()
"""

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid
from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):

        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0,
                             sd=1,
                             low=-rad,
                             upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes,
                          self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes,
                          self.no_of_hidden_nodes))

    def train(self, input_vector, target_vector):


        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.wih,
                                input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who,
                                output_hidden)
        output_network = activation_function(output_vector2)

        output_errors = target_vector - output_network
        # update the weights:
        tmp = output_errors * output_network \
              * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp,
                                          output_hidden.T)
        self.who += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T,
                               output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * \
              (1.0 - output_hidden)
        self.wih += self.learning_rate \
                    * np.dot(tmp, input_vector.T)

    def run(self, input_vector):
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.wih,
                               input_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.who,
                               output_vector)
        output_vector = activation_function(output_vector)

        return output_vector

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((26, 26), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm

    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


# ....................Train and Test..........................................
ANN = NeuralNetwork(no_of_in_nodes=image_pixels,
                    no_of_out_nodes=26,
                    no_of_hidden_nodes=100,
                    learning_rate=0.1)

for i in range(len(train_imgs)):
    ANN.train(train_imgs[i], train_labels_one_hot[i])
for i in range(20):
    res = ANN.run(test_imgs[i])
    print(test_labels[i], np.argmax(res), np.max(res))

corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
print("corrects and wrongs of train", corrects," ",wrongs)
print("accruracy train: ", float(corrects / (corrects + wrongs)))
corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
print("corrects and wrongs of test", corrects," ",wrongs)
print("accruracy: test", float(corrects / (corrects + wrongs)))
cm = ANN.confusion_matrix(train_imgs, train_labels)
print(cm)
for i in range(10):
    print("Alphabet: ", i, "precision: ", ANN.precision(i, cm), "recall: ", ANN.recall(i, cm))