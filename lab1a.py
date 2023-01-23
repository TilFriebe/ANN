import numpy as np
import matplotlib.pyplot as plt
def main():
    n, bias, sigmaW, epochs = 100, 0, 0.1, 5


    weights = np.random.standard_normal(3) * sigmaW

    #mean and standard deviation set A
    mA = [1, 0]
    sigmaA = 0.3

    #mean and standard deviation set B
    mB = [-3, 2]
    sigmaB = 0.3

    #create the sets, each containing n elements
    classA1 = np.random.standard_normal(n) * sigmaA + mA[0]
    classA2 = np.random.standard_normal(n) * sigmaA + mA[1]
    classB1 = np.random.standard_normal(n) * sigmaB + mB[0]
    classB2 = np.random.standard_normal(n) * sigmaB + mB[1]

    A, B = np.vstack((classA1, classA2)), np.vstack((classB1, classB2))

    #learning rates
    eta = [0.0001, 0.001, 0.01, 0.1]

    #all errors and weights for different eta
    allErrors = []
    allWeights = []

    learning_methods = [perceptron_learning, delta_learning_online, delta_learning_batch]

    #looping over all eta
    for i in eta:
        data = perceptron(A, B, n, weights, learning_methods, i, epochs)
        allWeights.append(data[0])
        allErrors.append(data[1])


    # plot graph
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')

    # Plot 1
    # calculating errors
    xList = [i for i in range(1, epochs + 1)]

    plt.subplot(1, 2, 2)
    plt.plot(xList, allErrors[0][0], xList, allErrors[0][1], xList, allErrors[0][2])
    plt.legend(("Perceptron learning", "Delta rule online", "Delta rule batch"))

    plt.ylabel('errors')
    plt.xlabel('epochs')

    # Plot 2
    plt.subplot(1, 2, 1)

    plt.plot(classA1, classA2, 'x', color="green")
    plt.plot(classB1, classB2, 'x', color="red")

    # to plot the initial hyperplane (red)
    plt.arrow((bias / np.dot(weights, weights)) * weights[0], (bias / np.dot(weights, weights)) * weights[1],
              weights[0], weights[1], head_width=0.2, color="red", linewidth=2.0)
    x = np.linspace(-4, 4)
    plt.plot(x, (bias - weights[0] * x) / weights[1], '-r', linewidth=1.5)

    plot_map = ["blue", "orange", "green", "--b", "--o", "--g"]

    # Hyperplane after epochs
    for i in range(len(learning_methods)):
        plt.arrow((bias / np.dot(allWeights[0][i], allWeights[0][i])) * allWeights[0][i][0],
                  (bias / np.dot(allWeights[0][i], allWeights[0][i])) * allWeights[0][i][1],
                  allWeights[0][i][0], allWeights[0][i][1], head_width=0.2, color=plot_map[i])
        x = np.linspace(-4, 4)
        plt.plot(x, (bias - allWeights[0][i][0] * x) / allWeights[0][i][1], plot_map[3 + i])




    plt.show()

def perceptron(classPos, classNeg, n, weights, learning_methods, eta, epochs):
    classes = np.vstack((np.hstack((classNeg, classPos)), np.array((2 * n) * [1])))
    allWeights = [weights.copy() for i in range(len(learning_methods))]
    allErrors = [[] for i in range(len(learning_methods))]

    for i in range(epochs):
        # deciding order of samples. Neg: 1 to n, Pos: n+1 to 2n
        orderSamples = np.random.permutation(2 * n)
        for j in range(len(learning_methods)):
            learning_methods[j](n, classes, allWeights[j], eta, orderSamples, allErrors[j])


    return [allWeights, allErrors]

def perceptron_learning(n, classes, weights, eta, permutation, error):
    totError = 0
    for i in permutation:
        t = i // n
        x = classes[:, i]

        y = threshold(np.dot(weights, x), weights[1])
        weights += eta * np.dot((t - y), x)
        totError += check_errors(weights, n, classes, i)
    error.append(totError)
    return [weights, error]

def delta_learning_online(n, classes, weights, eta, permutation, error):
    totError = 0
    for i in permutation:
        t = (i // n) * 2 - 1
        x = classes[:, i]
        y = np.dot(weights, classes[:, i])
        weights += eta * np.dot((t - y), x)
        totError += check_errors(weights, n, classes, i)
    error.append(totError)
    return [weights, error]

def delta_learning_batch(n, X, W, eta, permutation, error):
    totError = 0

    for i in permutation:
        totError += check_errors(W, n, X, i)
    T = np.array(n * [-1] + n * [1])
    W += eta * np.dot(T - np.dot(W, X), np.transpose(X))
    error.append(totError)
    return [W, error]



def threshold(y, bias):
    if y > bias:
        return 1
    return 0


def check_errors(W, n, X, i):
    error = 0
    norm = np.sqrt(np.dot(W, W))
    projOnW = (np.dot(X[:, i], W) / norm ** 2) * W
    if ((np.dot(projOnW, W) > 0 and i < n) or (np.dot(projOnW, W) <= 0 and i > n - 1)):
        error = 1
    return error

if __name__ == '__main__':
    main()
