import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([[10], [10]])
    b = np.array([-15])
    z = np.dot(x, w) + b
    y = sigmoid(z)
    return y

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([[10], [10]])
    b = np.array([-5])
    z = np.dot(x, w) + b
    y = sigmoid(z)
    return y

def XOR(x1, x2):
    x = np.array([x1, x2])
    w1 = np.array([[10, -10], [-10, 10]])
    b1 = np.array([-5, -5])
    w2 = np.array([[10], [10]])
    b2 = np.array([-5])
    z1 = np.dot(x, w1) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(h1, w2) + b2
    y = sigmoid(z2)
    return y


for i in range(2):
    for j in range(2):
        print("%d and %d = %f" % (i, j, AND(i, j)))
print()

for i in range(2):
    for j in range(2):
        print("%d or %d = %f" % (i, j, OR(i, j)))
print()

for i in range(2):
    for j in range(2):
        print("%d xor %d = %f" % (i, j, XOR(i, j)))
print()