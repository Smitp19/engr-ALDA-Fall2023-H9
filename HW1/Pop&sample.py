import math

X = [7, 12, 10, 9, 14, 10, 11]

population_mean = sum(X) / len(X)

population_SD = math.sqrt(sum([(xi - population_mean) ** 2 for xi in X]) / len(X))


sample_mean = sum(X) / (len(X)-1)

sample_SD = math.sqrt(sum([(xi-sample_mean) ** 2 for xi in X]) / (len(X)-1))
squared_X = [x**2 for x in X]
LHS_population = sum(squared_X) / len(X)
RHS_population = (population_mean ** 2) + (population_SD ** 2)


LHS_sample = sum(squared_X) / (len(X)-1)
RHS_sample = (sample_mean ** 2) + (sample_SD ** 2)

print("Population Standard Deviation (sigma(x)): ", population_SD)
print("Sample Standard Deviation (s(x)): ", sample_SD)
print("Verification for Population Standard Deviation: ¯x^2 = (¯x^2 + σ^2(x)): ", LHS_population == RHS_population)
print("Verification for Sample Standard Deviation: ¯x^2 = (¯x^2 + σ^2(x)): ", LHS_sample == RHS_sample)
