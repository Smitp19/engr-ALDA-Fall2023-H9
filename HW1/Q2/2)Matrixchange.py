import numpy as np

# Construct a 5*5 matrix of 1's using NumPy
A = np.matrix([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

# Changing all the Element in 3rd Row to 3
A[3] = (3, 3, 3, 3, 3)

# Print the Matrix
print(A)
