import numpy as np
import matplotlib.pyplot as plt


def main():
    n, bias, eta, epochs = 100, 1, [0.00001, 0.0001, 0.001, 0.01, 0.1], 30

    # mean and standard deviation set A
    mA = [0, 0]
    sigmaA = 0.9

    # mean and standard deviation set B
    mB = [-5, -3]
    sigmaB = 0.9

    '''
    Weights standard deviation, integers?

    '''
    # standard deviation Weights
    sigmaW = 0.01
    dimension = 2
    weights = InitialWeightMatrix(sigmaW)
    print("Initial weight matrix ", weights, "line k ", weights[0] / weights[1], "\n")

    # create the sets, each containing n elements
    #classA1 = np.random.standard_normal(n) * sigmaA + mA[0]
    # For 3.1.3
    classA1 = np.append(np.random.standard_normal(int(n/2)) * sigmaA + mA[0], np.random.standard_normal(int(n/2)) * sigmaA - mA[0])
    classA2 = np.random.standard_normal(n) * sigmaA + mA[1]
    classB1 = np.random.standard_normal(n) * sigmaB + mB[0]
    classB2 = np.random.standard_normal(n) * sigmaB + mB[1]

    A, B = np.vstack((classA1, classA2)), np.vstack((classB1, classB2))

    red_sample1 = remove_samples_randomly(A, B, n, 25, 25)
    red_sample2 = remove_samples_randomly(A, B, n, 50, 0)
    red_sample3 = remove_samples_randomly(A, B, n, 0,  50)
    red_sample4 = remove_samples_classA(A, n, 20, 80)
    sample5 = [red_sample4, B]
    samples = [red_sample1, red_sample2, red_sample3]
    # all errors and weights for different eta
    allErrors = []
    allWeights = []
    allMSE = []



    # looping over all eta
    #learning_methods = [perceptron_learning, delta_learning_online, delta_learning_batch]
    learning_methods= [perceptron_learning, delta_learning_batch]
    #classTest = Learning(n, A, B, T, eta)
    #newWeights = [[], [], []]
    for j in range(len(samples)):
        allErrors.append([])
        allWeights.append([])
        for i in eta:
            data = perceptron(samples[j][0], samples[j][1], 75, weights, learning_methods, epochs, i)
            allErrors[j].append(data[1])
            allWeights[j].append(data[0])
            #allMSE.append(data[2])
    #print("all errors ", allErrors, "\n")

    #print("print SHAPE OF ERRORS", np.shape(allErrors), "\n")
    #print("print allErrors before plot: ", allErrors)

    #plot_mapping = ["green", "red", "blue", "--g", "--r", "--b"]
    plot_mapping = ["green", "blue", "--g", "--b"]
    #plot_mapping = ["blue", "--b"]
    x = np.linspace(-10, 10)
    for k in range(len(samples)):
        for j in range(len(eta)):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Horizontally stacked subplots eta = ' + str(eta[j]))

            # Plot 1
            plt.subplot(1, 2, 1)
            plt.title('Visualization of the classification')
            plt.grid(True)
            plt.plot(samples[k][0][0], samples[k][0][1], 'x', color="green")
            plt.plot(samples[k][1][0], samples[k][1][1], 'x', color="red")
            plt.ylim([-10, 10])

            # to plot the initial hyperplane (red)
            plt.plot(x, -(weights[2] + weights[0] * x) / weights[1], color = 'black', linewidth=1.5)
            for i in range(len(learning_methods)):
                # Hyperplane after epochs
                plt.plot(x, -(allWeights[k][j][i][2] + allWeights[k][j][i][0] * x) / allWeights[k][j][i][1], plot_mapping[2 + i])

            # Plot 2
            plt.subplot(1, 2, 2)
            #plt.plot(allErrors[j][0], 'green', allErrors[j][1], 'red', allErrors[j][2], 'blue')
            plt.plot(allErrors[k][j][0], 'green', allErrors[k][j][1], 'blue',)
            plt.ylabel('Correct classifications')
            plt.xlabel('epochs')
            plt.title('Correct classifications per epoch')

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

    classes = np.vstack((np.hstack((classNeg, classPos)), np.array(2*n * [1])))
    #classes = np.hstack((classNeg, classPos))
    allWeights = [weights.copy() for i in range(len(learning_methods))]
    allErrors = [[] for i in range(len(learning_methods))]
    allMSE = [[] for i in range(len(learning_methods))]
    # add initial vaules to allErrors
    for i in range(len(learning_methods)):
        allErrors[i].append(nErrors(allWeights[i], n, classes))
        allMSE[i].append(MSE(allWeights[i], n, classes))
    for i in range(epochs):
        #Neg: 1 to n, Pos: n+1 to 2n
        orderSamples = np.random.permutation(2 * n)
        for j in range(len(learning_methods)):
            learning_methods[j](n, classes, allWeights[j], eta, orderSamples)
            allErrors[j].append(nErrors(allWeights[j], n, classes))
            allMSE[j].append(MSE(allWeights[j], n, classes))

    return [allWeights, allErrors, allMSE]


def perceptron_learning(n, X, W, eta, permutation):
    T = []
    for i in permutation:
        T.append(i // n)
        W += eta * np.dot(T[-1] - threshold(np.dot(W, X[:, i])), X[:, i])


def delta_learning_online(n, X, W, eta, permutation):
    T = []
    for i in permutation:
        T.append(((i // n) * 2) - 1)
        W += eta * (T[-1] - np.dot(np.transpose(W), X[:, i])) * np.transpose(X[:, i])


def delta_learning_batch(n, X, W, eta, permutation):
    T = []
    weights_temp = [0, 0, 0]
    #weights_temp = [0, 0]
    for i in permutation:
        T.append((i // n) * 2 - 1)
        weights_temp += eta * (T[-1] - np.dot(np.transpose(W), X[:, i])) * np.transpose(X[:, i])
    W += weights_temp


def nErrors(W, n, X):
    error = 0
    norm = np.sqrt(np.dot(W, W))
    for i in range(2 * n):
        projOnW = (np.dot(X[:, (i)], W) / norm ** 2) * W
        if ((np.dot(projOnW, W) > 0 and i < n) or (np.dot(projOnW, W) <= 0 and i > n - 1)):
            error += 1
    return (2*n - error)

def MSE(W, n, X):
    MSEerror=0
    norm = np.sqrt(np.dot(W, W))
    for i in range(2 * n):
        projOnW = (np.dot(X[:, (i)], W) / norm ** 2) * W
        projOnHyp = W - projOnW
        MSEerror += np.sum(np.sqrt((X[:, (i)]-projOnHyp)**2)) / n
    return MSEerror

def remove_samples_randomly(A, B, n, perc_A, perc_B):
    """
    :param A: dataset class A
    :param B: dataset class B
    :param perc_A: percentage of A to be removed
    :param perc_B: percentage of B to be removed
    :return: reduced datasets A_red and B_red
    """

    ### for (a): 25,
    idx_A = np.random.uniform(0, n, n - int(n*perc_A/100)).astype('int')
    idx_B = np.random.uniform(0, n, n - int(n*perc_B/100)).astype('int')

    A_red = A[:, idx_A]
    B_red = B[:, idx_B]

    return A_red, B_red
    

def remove_samples_classA(A, n, perc_A_1, perc_A_2):
    '''
    :param A: samples of class A
    :param n: 100
    :param perc_A_1: percentage to be removed from A<0
    :param perc_A_2: percentage to be removed from A>0
    :return: A_red: reduced samples of class A
    '''

    A_1 = A[:, A[1, :] < 0]
    A_2 = A[:, A[1, :] > 0]

    len_A_1 = np.shape(A_1)[1]
    len_A_2 = np.shape(A_2)[1]
    # n_A_1 = np.round(((n-perc_A_1)/n) * len_A_1)
    n_A_1 = np.round(((n - n*perc_A_1/100)/n) * len_A_1)
    # n_A_2 = np.round(((n-perc_A_2)/n) * len_A_2)
    n_A_2 = np.round(((n - n*perc_A_2/100)/n) * len_A_2)

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


def InitialWeightMatrix(sigmaW):

    return np.random.standard_normal(3) * sigmaW


if __name__ == '__main__':
    main()


'''
Epochs
Convergence
Bias delta rule
'''