from math import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut

# Load the dataset
data = pd.read_csv(r"C:\Users\smith\Desktop\NCSU-SEM-1\ALDA\HW\HW-4\adj_real_estate.csv")

X = data.drop(columns='Y house price of unit area')
x1 = data['X1 house age']
x2 = data['X2 distance to the nearest MRT station']
x3 = data['X3 number of convenience stores']

y = data['Y house price of unit area']


class linearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones for the intercept term
        ones_column = np.ones((X.shape[0], 1))
        X = np.hstack((ones_column, X))

        # Calculate coefficients using the closed-form solution (normal equation)
        X_transpose = np.transpose(X)
        X_transpose_X = X_transpose.dot(X)
        X_transpose_y = X_transpose.dot(y)
        self.coefficients = np.linalg.inv(X_transpose_X).dot(X_transpose_y)

    def predict(self, X):
        # Add a column of ones for the intercept term
        ones_column = np.ones((X.shape[0], 1))
        X = np.hstack((ones_column, X))

        # Make predictions
        predictions = X.dot(self.coefficients)
        return predictions


model1 = linearRegression()
model2 = linearRegression()
model3 = linearRegression()

# Fit the data to the three models
model1.fit(np.vstack((x1, x2, x3)).T, y)
model2.fit(np.vstack((x1 ** 2, x2 ** 2, x3 ** 2)).T, y)
model3.fit(np.vstack((x1 ** 3, x2 ** 3, x3 ** 3)).T, y)

# Get the coefficients for the three models
alpha_s = [model1.coefficients]
beta_s = [model2.coefficients]
gamma_s = [model3.coefficients]

# Print the coefficients
print("Coefficients for Model 1 (alpha_s):", alpha_s)
print("Coefficients for Model 2 (beta_s):", beta_s)
print("Coefficients for Model 3 (gamma_s):", gamma_s)

# Create LOOCV iterator
loo = LeaveOneOut()

# Initialize lists to store RMSE values
rmse_model1 = []
rmse_model2 = []
rmse_model3 = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit Model 1
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    rmse_model1.append(sqrt(np.mean((y_test - y_pred1) ** 2)))

    # Fit Model 2
    model2.fit(X_train ** 2, y_train)
    y_pred2 = model2.predict(X_test ** 2)
    rmse_model2.append(sqrt(np.mean((y_test - y_pred2) ** 2)))

    # Fit Model 3
    model3.fit(X_train ** 3, y_train)
    y_pred3 = model3.predict(X_test ** 3)
    rmse_model3.append(sqrt(np.mean((y_test - y_pred3) ** 2)))

# Calculate the mean RMSE for each model
mean_rmse_model1 = np.mean(rmse_model1)
mean_rmse_model2 = np.mean(rmse_model2)
mean_rmse_model3 = np.mean(rmse_model3)

# Print RMSE for the three models
print("RMSE for Model 1:", mean_rmse_model1)
print("RMSE for Model 2:", mean_rmse_model2)
print("RMSE for Model 3:", mean_rmse_model3)

# Identify the best model based on RMSE
best_model = min(
    [("Model 1", mean_rmse_model1), ("Model 2", mean_rmse_model2), ("Model 3", mean_rmse_model3)],
    key=lambda x: x[1]
)

print("The best-fitting model is:", best_model[0])
