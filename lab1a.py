import numpy as np
import matplotlib.pyplot as plt


def main():
    n, bias, eta, epochs = 100, 1, [0.01], 5

    # mean and standard deviation set A
    mA = [-5, 0]
    sigmaA = 1

    # mean and standard deviation set B
    mB = [5, 0]
    sigmaB = 1

    '''
    Weights standard deviation, integers?

    '''
    # standard deviation Weights
    sigmaW = 0.1
    dimension = 2
    weights = InitialWeightMatrix(dimension, sigmaW, bias)
    print("Initial weight matrix ", weights, "line k ", weights[0] / weights[1], "\n")

    # create the sets, each containing n elements
    classA1 = np.random.standard_normal(n) * sigmaA + mA[0]
    # For 3.1.3
    # classA1 = np.append(np.random.standard_normal(n/2) * sigmaA + mA[0], np.random.standard_normal(n/2) * sigmaA - mA[0])
    classA2 = np.random.standard_normal(n) * sigmaA + mA[1]
    classB1 = np.random.standard_normal(n) * sigmaB + mB[0]
    classB2 = np.random.standard_normal(n) * sigmaB + mB[1]

    A, B = np.vstack((classA1, classA2)), np.vstack((classB1, classB2))

    # all errors and weights for different eta
    allErrors = []
    allWeights = []



    # looping over all eta
    learning_methods = [perceptron_learning, delta_learning_online, delta_learning_batch]
    for i in eta:
        data = perceptron(A, B, n, weights, bias, learning_methods, epochs, i, False)
        allErrors.append(data[1])
        allWeights.append(data[0])
    print("all errors ", allErrors, "\n")
    weights1 = allWeights[0][0]

    print("print SHAPE OF ERRORS", np.shape(allErrors), "\n")
    print("print allErrors before plot: ", allErrors)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')

    # Plot 1
    error1 = allErrors[0][0]

    plt.subplot(1, 2, 2)
    plt.plot(error1, 'black')

    plt.ylabel('errors')
    plt.xlabel('epochs')

    # Plot 2
    plt.subplot(1, 2, 1)

    plt.plot(classA1, classA2, 'x', color="green")
    plt.plot(classB1, classB2, 'x', color="red")
    plt.ylim([-5, 5])

    # to plot the initial hyperplane (red)
    plt.arrow((bias / np.dot(weights, weights)) * weights[0], (bias / np.dot(weights, weights)) * weights[1],
              weights[0], weights[1], head_width=0.2, color="red", linewidth=2.0)
    x = np.linspace(-4, 4)
    plt.plot(x, -(bias + weights[0] * x) / weights[1], '-r', linewidth=1.5)

    # Hyperplane after epochs
    plt.arrow((weights1[2] * bias / np.dot(weights1, weights1)) * weights1[0], (weights1[2]*bias / np.dot(weights1, weights1)) * weights1[1],
              weights1[0], weights1[1], head_width=0.2, color="green")
    x = np.linspace(-4, 4)
    plt.plot(x, -(weights1[2] + weights1[0] * x) / weights1[1], '--b')

    plt.show()


def perceptron(classPos, classNeg, n, weights, bias, learning_methods, epochs, eta=1, batch=True):
    """

    :param classPos1: x values for class 1
    :param classPos2: y values for class 1
    :param classNeg1: x values for class 2
    :param classNeg2: y values for class 2
    :param n: number of data points in each class
    :param weights: weights of the linear line
    :param bias: A shifting parameter to move it away from origin
    :param learning_method: choose either perceptron learning or delta rule
    :param eta: learning rate
    :param batch: choose either batch learning or on-line earning
    :return: weights and errors
    """
    #
    classes = np.vstack((np.hstack((classNeg, classPos)), np.array((2 * n) * [1])))
    allWeights = [weights.copy() for i in range(len(learning_methods))]
    allErrors = [[] for i in range(len(learning_methods))]

    # add initial vaules to allErrors
    for i in range(len(learning_methods)):
        allErrors[i].append(nErrors(allWeights[i], n, classes))

    for i in range(epochs):
        # deciding order of samples. Neg: 1 to n, Pos: n+1 to 2n
        orderSamples = np.random.permutation(2 * n)
        for j in range(len(learning_methods)):
            variables = learning_methods[j](n, classes, allWeights[j], eta, orderSamples)
            allWeights[j] = variables[0]
            allErrors[j].append(nErrors(variables[0], n, classes))

    return [allWeights, allErrors]


def perceptron_learning(n, X, W, eta, permutation):
    T = []
    for i in permutation:
        T.append(i // n)
        W += eta * np.dot(T[-1] - threshold(np.dot(W, X[:, i])), X[:, i])
        # print("T for ",i+1, " ", T[-1])
        # print(threshold(np.dot(W, X[:, (i - 1)])))
        # print(T[-1] - threshold(np.dot(W, X[:, i])))
    return [W, np.array(T)]


def delta_learning_online(n, X, W, eta, permutation):
    T = []
    for i in permutation:
        T.append((i // n) * 2 - 1)
        W += eta * np.dot(T[-1] - np.dot(W, X[:, i]), X[:, i])
        print(T[-1] - np.dot(W, X[:, i]))
    return [W, np.array(T)]


def delta_learning_batch(n, X, W, eta, permutation):
    T = np.array(n * [-1] + n * [1])
    W += eta * np.dot(T - np.dot(W, X), np.transpose(X))
    return W, T


def nErrors(W, n, X):
    error = 0
    norm = np.sqrt(np.dot(W, W))
    for i in range(2 * n):
        projOnW = (np.dot(X[:, (i)], W) / norm ** 2) * W
        if ((np.dot(projOnW, W) > 0 and i < n) or (np.dot(projOnW, W) <= 0 and i > n - 1)):
            error += 1
    return error

    """

    :param n: number of data points in each class
    :param i: index of a certain Data point
    :param classes: A list of classes including their coordinates
    :param weights: weights of the linear line
    :param bias: A shifting parameter to move it away from origin
    :return: The target vakue t, the coordinates for the data point x and the estimated target value y
    """

    """

    :param n: number of data points in each class
    :param i: index of a certain Data point
    :param classes: A list of classes including their coordinates
    :param weights: weights of the linear line
    :param bias: A shifting parameter to move it away from origin
    :return: The target vakue t, the coordinates for the data point x and the estimated target value y
    """


def threshold(y):
    """

    :param y: A number representing whether a data point is in the correct classification
    :param bias: A shifting parameter to move it away from origin
    :return: The estimated target value
    """
    if y > 0:
        return 1
    return 0


def InitialWeightMatrix(dimension, sigmaW, bias):

    return np.random.standard_normal(3) * sigmaW


if __name__ == '__main__':
    main()


'''
Epochs
Convergence
Bias delta rule
'''