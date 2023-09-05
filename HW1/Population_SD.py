import math

X = [7, 12, 10, 9, 14, 10, 11]

population_mean = sum(X) / len(X)

population_SD = math.sqrt(sum([(xi - population_mean) ** 2 for xi in X]) / len(X))

squared_X = [x**2 for x in X]

LHS_population = sum(squared_X) / len(X)

RHS_population = (population_mean ** 2) + (population_SD ** 2)

print("Population Standard Deviation (sigma(x)): ", population_SD)

print("Verification for Population Standard Deviation: ¯x^2 = (¯x^2 + σ^2(x)): ", LHS_population == RHS_population)
