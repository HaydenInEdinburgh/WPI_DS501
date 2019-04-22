import numpy as np

A = np.mat([[0., 1., 1.],
                [1., 0., 0.],
                [1., 1., 0.]])

alpha = 0.85

sum_A = np.sum(A, axis=0)
print sum_A
n = A.shape[0]
print n
S = np.ones((n,n),float) / n
print S
P = A / sum_A
print P
G = (alpha * P ) + (1- alpha)*S
print G
print type(A)
print type(sum_A)