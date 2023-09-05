import numpy as np

# Constructing the Matrices
X = np.array([1, 2, 3, 4])
Y = np.array([1, 2, 4, 8])
Z = np.array([7, 3, 5, 1])

# Calculating the covariance of the matrices
cov_matrix = np.cov([X, Y, Z])

# Print covariance
print('covariance of the matrices = ', cov_matrix)

# calculating the Pearson correlation coefficient of the Matrix X and Y
cov_xy = cov_matrix[0, 1]
SD_X = np.std(X)
SD_Y = np.std(Y)
correlation_coefficient = cov_xy / (SD_X * SD_Y)

# Print Coefficient
print('Pearson correlation coefficient of the matrices = ', correlation_coefficient)
