import numpy as np
np.random.seed(2023)

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

# Calculating Standard Deviation of Transpose Matrix
SD = np.std(Transpose, axis=1)

# Converting the array
SDNew = np.transpose(SD)

# Uniform random Number between 0 and 1
randomNumbers = np.random.rand(6)

# Stacking the SD and randomNumbers to form matrix B
B = np.vstack((SDNew, randomNumbers))

# Print the  Matrix B
print(B)
