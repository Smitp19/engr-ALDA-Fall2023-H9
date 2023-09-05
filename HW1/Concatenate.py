import numpy as np

# Construct a 5*5 matrix of 1's using NumPy
A = np.matrix([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

# Changing all the Element in 3rd Row to 3
A[3] = (3, 3, 3, 3, 3)

# Sum of each Row of the Matrix
SumofRows = A.sum(axis=1)

# Concatenating of Matrix A and SumofRows
Concatenate = np.column_stack((A, SumofRows))

# Transpose of matrix A
Transpose = np.transpose(Concatenate)

# Print the Concatenated Matrix
print(Transpose)
