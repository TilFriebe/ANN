import numpy as np
import matplotlib.pyplot as plt
def main():
    n = 100
    mA =[1, 0.3]
    sigmaA = 0.2
    mB = [0, -0.1]
    sigmaB = 0.3
    classA1 = [np.random.standard_normal(n//2) * sigmaA - mA[0], np.random.standard_normal(n//2) * sigmaA + mA[0]]
    classA2 = np.random.standard_normal(n) * sigmaA + mA[1]
    classB1 = np.random.standard_normal(n) * sigmaB + mB[0]
    classB2 = np.random.standard_normal(n) * sigmaB + mB[1]
    pass

if __name__ == '__main__':
    main()