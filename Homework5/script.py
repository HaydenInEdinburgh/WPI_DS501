import numpy as np
s = np.array([[0, 1, 1],
              [1, 0, -1],
              [1, 1, 0]])
r, c = np.where(s == 0)
valid_index = list(zip(r, c))
i = np.random.choice(valid_index)
print(i)
