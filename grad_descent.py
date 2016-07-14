# Author :   Tiancheng Gong
# Date   :   07/13/2016
# Version:   1.0

import csv
import random
import numpy as np


# retrieve data from 'filename', and return them as x and y
# x   : n * k, k is the number of features
# y   : n * 1
def retrieve_data(filename):
    with open(filename + '.csv', 'rb') as data:
        x = []
        y = []
        reader = csv.reader(data, delimiter=',')
        for row in reader:
            x.append(np.asarray(row[0: -1], dtype=np.float64))
            y.append(np.asarray(row[-1], dtype=np.float64))
    return x, y


# randomly divide the dataset into training dataset and testing dataset
# according to the  scale, and return training_x, training_y, testing_x
# and testing_y.
# training   : n * scale
# testing    : n * (1 - scale)
def divide_dataset(x, y, scale):
    size = len(x)
    train_set = random.sample(range(size), int(scale * size))
    training_x = [x[i] for i in train_set]
    training_y = [y[i] for i in train_set]
    testing_x = [e for i, e in enumerate(x) if i not in train_set]
    testing_y = [e for i, e in enumerate(y) if i not in train_set]
    return training_x, training_y, testing_x, testing_y


def setup(length):
    res = np.zeros(length + 1)
    return res


def calculate_cost(x, y, theta):
    arg_x = np.insert(x, 0, 1)
    exp = np.dot(arg_x, theta)
    error = y - exp
    return error * arg_x


def prepare_data(filename, scale):
    x, y = retrieve_data('dataset1')
    return divide_dataset(x, y, scale)


def train(tr_x, tr_y, lr, numOfIter):
    numOfTesting = len(tr_x)
    theta = setup(len(tr_x[0]))
    mean_abs_err = 0
    for i in range(numOfIter):
        err = 0
        for j in range(numOfTesting):
            arg_x = np.insert(tr_x[j], 0, 1)
            exp = np.dot(arg_x, theta)
            delta = tr_y[j] - exp
            err += abs(delta)
            for k in range(len(theta)):
                theta[k] += lr * delta * arg_x[k]
        print i
        mean_abs_err = err / numOfTesting
    return theta, mean_abs_err




if __name__ == '__main__':
    scale = 0.8  # the scale of training data and testing data
    lr = 0.0001  # the learning rate
    numOfIter = 1500  # the number of iterations using the lr

    tr_x, tr_y, te_x, te_y = prepare_data('dataset1', scale)

    theta, mean_abs_err = train(tr_x, tr_y, lr, numOfIter)


    result = open('result.txt', 'w')
    count = 0
    result.write('theta:')
    for i in range(len(theta)):
        result.write(' ' + str(theta[i]))
    result.write('\n')
    for i in range(len(te_x)):
        arg_x = np.insert(te_x[i], 0, 1)
        exp = np.dot(arg_x, theta)
        err = te_y[i] - exp
        pct = abs(err / te_y[i]) * 100
        count += pct
        result.write('exact: '+ str(te_y[i]) + ' exp: ' + str(exp) + ' err: ' + str(err) + ' err_pct: ' + str(pct) + '\n')
    result.write('Size of tesing dataset: ' + str(len(te_x)) + '\n' + 'Total err_pct: ' + str(count / len(te_x)))
    result.close()
