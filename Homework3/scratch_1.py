import numpy as np

dt = np.loadtxt(open('movielens_test.csv'), delimiter=',')
for i in range(len(dt)):
    if dt[i][0] <= dt[i + 1][0]:
        n = dt[i + 1][0]
    if dt[i][1] <= dt[i][1]:
        m = dt[i + 1][1]
print dt