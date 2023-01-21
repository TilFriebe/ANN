import numpy as np
import matplotlib.pyplot as plt
def main():
    n, bias, eta, epochs = 100, 0, [0.03], 100

    #mean and standard deviation set A
    mA = [-3, -3]
    sigmaA = 0.3

    #mean and standard deviation set B
    mB = [3, 3]
    sigmaB = 0.3

    '''
    Weights standard deviation, integers?
    
    '''
    #standard deviation Weights
    sigmaW = 4
    dimension = 2
    weights = InitialWeightMatrix(dimension, sigmaW, bias)

    #create the sets, each containing n elements
    classA1 = np.random.standard_normal(n) * sigmaA + mA[0]
    # For 3.1.3
    # classA1 = np.append(np.random.standard_normal(n/2) * sigmaA + mA[0], np.random.standard_normal(n/2) * sigmaA - mA[0])
    classA2 = np.random.standard_normal(n) * sigmaA + mA[1]
    classB1 = np.random.standard_normal(n) * sigmaB + mB[0]
    classB2 = np.random.standard_normal(n) * sigmaB + mB[1]

    A, B = np.vstack((classA1, classA2)), np.vstack((classB1, classB2))

    #all errors and weights for different eta
    allErrors = []
    allWeights = []

    #plot graph
    plt.figure()


    plt.plot(classA1, classA2, 'x')
    plt.plot(classB1, classB2, 'x')

    #to plot the initial hyperplane (red)
    plt.arrow((bias/np.dot(weights,weights))*weights[0], (bias/np.dot(weights,weights))*weights[1], weights[0], weights[1], head_width=0.2, color="red")
    x=np.linspace(-4,4)
    plt.plot(x,(bias-weights[0]*x)/weights[1], '-r')

    #looping over all eta
    learning_methods = [perceptron_learning, delta_learning_online, delta_learning_batch]
    for i in eta:
        data = perceptron(A, B, n, weights, bias, learning_methods, epochs, i, False)
        allErrors.append(data[1])
        allWeights.append(data[0])

    weights = allWeights[0][0]
    weights1 = allWeights[0][1]
    weights2 = allWeights[0][2]



    """
    #calculating errors
    xList = [i for i in range(1, 2*n+1)]
    for i in range(2*n):
        #allErrors[0][i] -= 7
        #allErrors[1][i] -= 14
        #allErrors[2][i] -= 21
        pass

    plt.plot(xList[0:100], allErrors[0][0:100], xList[0:100], allErrors[1][0:100], xList[0:100], allErrors[2][0:100], xList[0:100], allErrors[3][0:100])
    plt.legend(("eta = 0.001", "eta = 0.01", "eta = 0.1", "eta = 1"))

    """
    plt.arrow((bias / np.dot(weights, weights)) * weights[0], (bias / np.dot(weights, weights)) * weights[1],
              weights[0], weights[1], head_width=0.2, color="green")
    x = np.linspace(-4, 4)
    plt.plot(x, (bias - weights[0] * x) / weights[1], '-g')

    plt.arrow((bias / np.dot(weights1, weights1)) * weights1[0], (bias / np.dot(weights1, weights1)) * weights1[1],
              weights1[0], weights1[1], head_width=0.2, color="blue")
    x = np.linspace(-4, 4)
    plt.plot(x, (bias - weights1[0] * x) / weights1[1], '-b')

    plt.arrow((bias / np.dot(weights2, weights2)) * weights2[0], (bias / np.dot(weights2, weights2)) * weights2[1],
              weights2[0], weights2[1], head_width=0.2, color="black")
    x = np.linspace(-4, 4)
    plt.plot(x, (bias - weights2[0] * x) / weights2[1], 'black')
    #To here

    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.axis('square')
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)


    plt.title('Classification of two datasets using classical perceptron learning')

    plt.show()

def perceptron(classPos, classNeg, n, weights, bias, learning_methods, epochs, eta = 1, batch = True):
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
    classes = np.vstack((np.hstack((classNeg, classPos)), np.array((2*n)*[1])))
    allWeights = [weights.copy() for i in range(len(learning_methods))]
    allErrors = [[] for i in range(len(learning_methods))]
    print(allWeights)


    for i in range(epochs):
        # deciding order of samples. Neg: 1 to n, Pos: n+1 to 2n
        orderSamples = np.random.permutation(2 * n)
        for j in range(len(learning_methods)):
            variables = learning_methods[j](n, classes, allWeights[j], eta, orderSamples)
            allWeights[j] = variables[0]
            allErrors[j].append(nErrors(variables[0], variables[1], classes))
    print(allWeights)
    '''
    for i in orderSamples:
        #t is the real value of the sample, x is the sample
        variables = learning_method(n, i, classes, weights, bias)
        t = variables[0]
        x = variables[1]
        y = variables[2]
        #update weights
        errors.append(t-y)
        if batch:
            weights_temp = np.add(weights_temp, np.dot(t-y, x))
        else:
            weights=np.add(weights,eta*np.dot(t-y,x))
    if batch:
        weights += eta*weights_temp
    '''
    # to plot the new hyperplanes (green)
    """
    plt.arrow((bias / np.dot(weights, weights)) * weights[0], (bias / np.dot(weights, weights)) * weights[1],
              weights[0], weights[1], head_width=0.2, color="green")
    x = np.linspace(-4, 4)
    plt.plot(x, (bias - weights[0] * x) / weights[1], '-g')
    """

    return [allWeights, allErrors]

def perceptron_learning(n, i, classes, weights, bias):
    """

    :param n: number of data points in each class
    :param i: index of a certain Data point
    :param classes: A list of classes including their coordinates
    :param weights: weights of the linear line
    :param bias: A shifting parameter to move it away from origin
    :return: The target vakue t, the coordinates for the data point x and the estimated target value y
    """
    t = (i - 1) // n
    x = [classes[t][0][(i - 1) % n], classes[t][1][(i - 1) % n]]
    y = threshold(np.dot(weights, x), bias)
    return [t, x, y]

def delta_rule(n, i, classes, weights, bias):
    """

    :param n: number of data points in each class
    :param i: index of a certain Data point
    :param classes: A list of classes including their coordinates
    :param weights: weights of the linear line
    :param bias: A shifting parameter to move it away from origin
    :return: The target vakue t, the coordinates for the data point x and the estimated target value y
    """
    t=((i-1)//n)*2-1
    if i > n:
        classPick = 0
    else:
        classPick = 1
    x = [classes[classPick][0][(i - 1) % n], classes[classPick][1][(i - 1) % n]]
    y = np.dot(weights, x)
    return [t, x, y]


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
    return np.append(np.random.standard_normal(dimension) * sigmaW,[-bias])

if __name__ == '__main__':
    main()


'''
Epochs
Convergence
Bias delta rule
'''