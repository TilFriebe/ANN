import numpy as np
import matplotlib.pyplot as plt


def main():
    n, bias, eta, epochs = 100, 1, [0.0001, 0.001, 0.01, 0.1], 50

    mA = [1, 0.3]
    sigmaA = 0.2

    # mean and standard deviation set B
    mB = [0, -0.1]
    sigmaB = 0.3

    # # mean and standard deviation set A
    # mA = [3, -2]
    # sigmaA = 0.9
    #
    # # mean and standard deviation set B
    # mB = [-1, 2]
    # sigmaB = 0.9

    '''
    Weights standard deviation, integers?

    '''
    # standard deviation Weights
    sigmaW = 0.01
    dimension = 2
    weights = InitialWeightMatrix(dimension, sigmaW, bias)
    print("Initial weight matrix ", weights, "line k ", weights[0] / weights[1], "\n")

    # create the sets, each containing n elements
    # classA1 = np.random.standard_normal(n) * sigmaA + mA[0]
    # For 3.1.3
    classA1 = np.append(np.random.standard_normal(int(n / 2)) * sigmaA + mA[0],
                        np.random.standard_normal(int(n / 2)) * sigmaA - mA[0])
    classA2 = np.random.standard_normal(n) * sigmaA + mA[1]
    classB1 = np.random.standard_normal(n) * sigmaB + mB[0]
    classB2 = np.random.standard_normal(n) * sigmaB + mB[1]

    A, B = np.vstack((classA1, classA2)), np.vstack((classB1, classB2))

    A = remove_samples_classa(A, n, 20, 80)
    # A, B = remove_samples_randomly(A, B, 0, 0)

    # all errors and weights for different eta
    allErrors = []
    allWeights = []
    allMSE = []

    # looping over all eta
    learning_methods = [perceptron_learning, delta_learning_online, delta_learning_batch]
    # classTest = Learning(n, A, B, T, eta)
    # newWeights = [[], [], []]
    for i in eta:
        data = perceptron(A, B, n, weights, learning_methods, epochs, i)
        allErrors.append(data[1])
        allWeights.append(data[0])
        # allMSE.append(data[2])
    print("all errors ", allErrors, "\n")
    print(allWeights)

    print("print SHAPE OF ERRORS", np.shape(allErrors), "\n")
    print("print allErrors before plot: ", allErrors)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # fig.suptitle('Horizontally stacked subplots')

    # Plot 2

    plt.subplot(1, 2, 2)
    plt.plot(allErrors[0][0], 'green', allErrors[0][1], 'red', allErrors[0][2], 'blue', )

    plt.ylabel('errors')
    plt.xlabel('epochs')
    """
    # Plot 3

    plt.subplot(1, 3, 3)
    plt.plot(allMSE[0][0], 'green', allMSE[0][1], 'red')


    plt.ylabel('errors')
    plt.xlabel('epochs')
    """
    # Plot 1
    plt.subplot(1, 2, 1)
    plt.grid(True)

    plt.plot(A[0, :], A[1, :], 'x', color="green")
    # plt.plot(classA1, classA2, 'x', color="green")
    plt.plot(B[0, :], B[1, :], 'x', color="red")
    # plt.plot(classB1, classB2, 'x', color="red")
    plt.ylim([-1.5, 1.5])
    plt.xlim([-2, 2])

    # to plot the initial hyperplane (red)
    plt.arrow((weights[2] / np.dot(weights, weights)) * weights[0],
              (weights[2] / np.dot(weights, weights)) * weights[1],
              weights[0], weights[1], head_width=0.2, color="red", linewidth=2.0)
    x = np.linspace(-4, 4)
    plt.plot(x, -(weights[2] + weights[0] * x) / weights[1], '-r', linewidth=1.5)
    plot_mapping = ["green", "red", "blue", "--g", "--y", "--b"]

    for i in range(len(learning_methods)):
        # Hyperplane after epochs
        x = np.linspace(-4, 4)
        plt.plot(x, -(allWeights[0][i][2] + allWeights[0][i][0] * x) / allWeights[0][i][1], plot_mapping[3 + i])

    ax2.legend([f'perceptron', f'$\Delta$ sequential', f'$\Delta$ batch'],
               loc='upper right')  # , bbox_to_anchor=(1, 0.5))
    # ax.legend([f'IQ phase {str(round(objs[0].phi_I_hum_sens))}°'], loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.get_legend().set_title(
        f'Learning Methods')

    plt.show()


def perceptron(classPos, classNeg, n, weights, learning_methods, epochs, eta=1):
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
    classes = np.vstack(
        (np.hstack((classNeg, classPos)), np.array((np.shape(classPos)[1] + np.shape(classNeg)[1]) * [1])))
    # classes = np.vstack((np.hstack((classNeg, classPos)), np.array((2 * n) * [1])))
    allWeights = [weights.copy() for i in range(len(learning_methods))]
    allErrors = [[] for i in range(len(learning_methods))]
    allMSE = [[] for i in range(len(learning_methods))]
    # add initial vaules to allErrors
    for i in range(len(learning_methods)):
        allErrors[i].append(nErrors(allWeights[i], n, classes, classNeg))
        allMSE[i].append(MSE(allWeights[i], n, classes, classNeg))
    for i in range(epochs):
        # Neg: 1 to n, Pos: n+1 to 2n
        orderSamples = np.random.permutation(np.shape(classes)[1])
        # orderSamples = np.random.permutation(2 * n)
        for j in range(len(learning_methods)):
            learning_methods[j](n, classes, classNeg, allWeights[j], eta, orderSamples)
            allErrors[j].append(nErrors(allWeights[j], n, classes, classNeg))
            allMSE[j].append(MSE(allWeights[j], n, classes, classNeg))

    return [allWeights, allErrors, allMSE]


def perceptron_learning(n, X, B, W, eta, permutation):
    # def perceptron_learning(n, X, W, eta, permutation):
    T = []
    for i in permutation:
        T.append(i // np.shape(B)[1])
        # T.append(i // n)
        W += eta * np.dot(T[-1] - threshold(np.dot(W, X[:, i])), X[:, i])


def delta_learning_online(n, X, B, W, eta, permutation):
    # def delta_learning_online(n, X, W, eta, permutation):
    T = []
    for i in permutation:
        T.append(((i // np.shape(B)[1]) * 2) - 1)
        # T.append(((i // n) * 2) - 1)
        W += eta * (T[-1] - np.dot(np.transpose(W), X[:, i])) * np.transpose(X[:, i])


def delta_learning_batch(n, X, B, W, eta, permutation):
    # def delta_learning_batch(n, X, W, eta, permutation):
    T = []
    weights_temp = [0, 0, 0]
    for i in permutation:
        T.append((i // np.shape(B)[1]) * 2 - 1)
        # T.append((i // n) * 2 - 1)
        weights_temp += eta * (T[-1] - np.dot(np.transpose(W), X[:, i])) * np.transpose(X[:, i])
    W += weights_temp


def nErrors(W, n, X, B):
    error = 0
    norm = np.sqrt(np.dot(W, W))
    for i in range(np.shape(X)[1]):
        # for i in range(2 * n):
        projOnW = (np.dot(X[:, (i)], W) / norm ** 2) * W
        if ((np.dot(projOnW, W) > 0 and i < np.shape(B)[1]) or (np.dot(projOnW, W) <= 0 and i > np.shape(B)[1] - 1)):
            # if ((np.dot(projOnW, W) > 0 and i < n) or (np.dot(projOnW, W) <= 0 and i > n - 1)):
            error += 1
    return error


def MSE(W, n, X, B):
    MSEerror = 0
    norm = np.sqrt(np.dot(W, W))
    for i in range(np.shape(X)[1]):
        # for i in range(2 * n):
        projOnW = (np.dot(X[:, (i)], W) / norm ** 2) * W
        projOnHyp = W - projOnW
        MSEerror += np.sum(np.sqrt((X[:, (i)] - projOnHyp) ** 2)) / n
    return MSEerror

    """

    :param n: number of data points in each class
    :param i: index of a certain Data point
    :param classes: A list of classes including their coordinates
    :param weights: weights of the linear line
    :param bias: A shifting parameter to move it away from origin
    :return: The target vakue t, the coordinates for the data point x and the estimated target value y
    """


def remove_samples_randomly(A, B, perc_A, perc_B):
    """
    :param A: dataset class A
    :param B: dataset class B
    :param perc_A: percentage of A to be removed
    :param perc_B: percentage of B to be removed
    :return: reduced datasets A_red and B_red
    """

    ### for (a): 25,
    idx_A = np.random.uniform(0, 100, 100 - perc_A).astype('int')
    idx_B = np.random.uniform(0, 100, 100 - perc_B).astype('int')

    A_red = A[:, idx_A]
    B_red = B[:, idx_B]

    return A_red, B_red


def remove_samples_classa(A, n, perc_A_1, perc_A_2):
    A_1 = A[:, A[0, :] < 0]
    A_2 = A[:, A[0, :] > 0]

    len_A_1 = np.shape(A_1)[1]
    len_A_2 = np.shape(A_2)[1]
    n_A_1 = np.round(((n - perc_A_1) / n) * len_A_1)
    n_A_2 = np.round(((n - perc_A_2) / n) * len_A_2)
    idx_A_1 = np.random.uniform(0, len_A_1, int(n_A_1)).astype('int')
    idx_A_2 = np.random.uniform(0, len_A_2, int(n_A_2)).astype('int')

    A_red1 = A_1[:, idx_A_1]
    A_red2 = A_2[:, idx_A_2]

    A_red = np.concatenate((A_red1, A_red2), axis=1)

    return A_red


def threshold(y):
    """

    :param y: A number representing whether a data point is in the correct classification
    :param bias: A shifting parameter to move it away from origin
    :return: The estimated target value
    """
    if y > 0:
        return 1
    return 0


def threshold_delta(y):
    """

    :param y: A number representing whether a data point is in the correct classification
    :param bias: A shifting parameter to move it away from origin
    :return: The estimated target value
    """
    if y > 0:
        return 1
    return -1


def InitialWeightMatrix(dimension, sigmaW, bias):
    return np.random.standard_normal(3) * sigmaW


if __name__ == '__main__':
    main()

'''
Epochs
Convergence
Bias delta rule
'''
