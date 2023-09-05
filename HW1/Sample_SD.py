import math

X = [7, 12, 10, 9, 14, 10, 11]

sample_mean = sum(X) / (len(X) - 1)

sample_SD = math.sqrt(sum([(xi - sample_mean) ** 2 for xi in X]) / (len(X) - 1))

squared_X = [x ** 2 for x in X]

LHS_sample = sum(squared_X) / (len(X) - 1)

RHS_sample = (sample_mean ** 2) + (sample_SD ** 2)
