import numpy as np
import matplotlib.pyplot as plt
def main():
    n = 100

    mA = [3, 0]
    sigmaA = 0.3

    mB = [-3, 0]
    sigmaB = 0.3
    classA1 = np.random.standard_normal(n) * sigmaA + mA[0]
    classA2 = np.random.standard_normal(n) * sigmaA + mA[1]
    classB1 = np.random.standard_normal(n) * sigmaB + mB[0]
    classB2 = np.random.standard_normal(n) * sigmaB + mB[1]
    w = perceptron(classA1, classA2, classB1, classB2, n)
    x = [-0.5, -0,3, -0.1, 0.1, 0.3, 0.5]
    print(w)
    y = [-w[0]/w[1] * i for i in x]
    plt.figure()
    plt.plot(x, y)
    plt.plot(classA1, classA2, 'x')
    plt.plot(classB1, classB2, 'x')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.show()
    #plt.figure()
    #plt.ylabel('x2')
    #plt.xlabel('x1')
    #plt.show()

def perceptron(classA1, classA2, classB1, classB2, n, b = 0):
    w = np.array([0.5, 0.7])
    rndSeq = np.random.permutation(2 * n)
    allSetsX = np.concatenate((classA1, classB1))
    allSetsY = np.concatenate((classA2, classB2))
    target = []
    current_target = 0
    eta = 0.0001
    for i in rndSeq:
        y = w[0] * allSetsX[i] + w[1] * allSetsY[i]
        var_threshold = threshold(y, b)
        if i > b:
            target.append(0)
        else:
            current_target = 1
            target.append(1)
        if var_threshold == current_target:
            continue
        else:
            if var_threshold == 0:
                delta_w = np.array([eta*allSetsX[i], eta*allSetsY[i]])
            else:
                delta_w = np.array([-eta*allSetsX[i], -eta*allSetsY[i]])
            w = w + delta_w
    return w


def threshold(y, b):
    if y > b:
        return 1
    return 0


if __name__ == '__main__':
    main()
