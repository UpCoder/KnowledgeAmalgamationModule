import numpy as np

a = np.reshape(range(8), (2, 2, 2))

b = a[:, :, 0]/a[:, :, 1]

print(b)