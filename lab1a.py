import numpy as np
import matplotlib.pyplot as plt
def main():
    n = 100
    weights = [-2, 0.7]
    bias = 0

    #mean and standard deviation set A
    mA = [-3, 0]
    sigmaA = 0.3

    #mean and standard deviation set B
    mB = [3, 0]
    sigmaB = 0.3

    #create the sets, each containing n elements
    classA1 = np.random.standard_normal(n) * sigmaA + mA[0]
    classA2 = np.random.standard_normal(n) * sigmaA + mA[1]
    classB1 = np.random.standard_normal(n) * sigmaB + mB[0]
    classB2 = np.random.standard_normal(n) * sigmaB + mB[1]

    #learning rates
    eta = [0.00001]

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
    for i in eta:
        data = perceptron(classA1, classA2, classB1, classB2, n, weights, bias, perceptron_learning, i)
        allErrors.append(data[1])
        allWeights.append(data[0])

    weights = allWeights[0]
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

    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.axis('square')
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)


    plt.title('Classification of two datasets using classical perceptron learning')

    plt.show()

def perceptron(classPos1, classPos2, classNeg1, classNeg2, n, weights, bias, learning_method, eta = 1, batch = True):
    classes=[[classNeg1, classNeg2], [classPos1, classPos2]]
    #deciding order of samples. Neg: 1 to n, Pos: n+1 to 2n
    orderSamples = np.random.permutation(2 * n)
    errors = []
    weights_temp = np.array([0, 0])
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
        weights = eta*weights_temp
    # to plot the new hyperplanes (green)
    plt.arrow((bias / np.dot(weights, weights)) * weights[0], (bias / np.dot(weights, weights)) * weights[1],
              weights[0], weights[1], head_width=0.2, color="green")
    x = np.linspace(-4, 4)
    plt.plot(x, (bias - weights[0] * x) / weights[1], '-g')

    return [weights, errors]

def perceptron_learning(n, i, classes, weights, bias):
    t = (i - 1) // n
    x = [classes[t][0][(i - 1) % n], classes[t][1][(i - 1) % n]]
    y = threshold(np.dot(weights, x), bias)
    return [t, x, y]

def delta_rule(n, i, classes, weights, bias):
    classPick = 0
    if i > n:
        t = 1
    else:
        t = -1
        classPick = 1
    x = [classes[classPick][0][(i - 1) % n], classes[classPick][1][(i - 1) % n]]
    y = np.dot(weights, x)
    return [t, x, y]


def threshold(y, bias):
    if y > bias:
        return 1
    return 0

if __name__ == '__main__':
    main()
