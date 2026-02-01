"""
This code implements several stochastic approximation algorithms introduced by Chapter 6
"""

import numpy as np

# 1. Mean Estimation

x = np.array([1*np.random.normal() for _ in range(1000)]) + 10 # Sample data
# mean
mean = np.sum(x) / len(x)
# incremental mean

alpha = lambda n: 5 / n
# Here I set alpha = 5/n, which is a valid choice since sum alpha_n = inf and sum alpha_n^2 < inf

inc_mean = x[0]
for k in range(1, len(x)+1):
    inc_mean = inc_mean - alpha(k) * (inc_mean - x[k-1])
print("Mean:", mean)
print("Incremental Mean:", inc_mean)


# 2. RM Algorithm for Root Finding

def f(x):
    return np.tanh(x) + x - 2

def f_noisy(x):
    noise = np.random.normal(scale=0.5)
    return f(x) + noise

# RM algorithm
x_rm = 0.5  # Initial guess
alpha = lambda n: 1 / n
for k in range(1, 100001):
    x_rm = x_rm - alpha(k) * f_noisy(x_rm)

print("Root found by RM:", x_rm)
print("Actual root:", 1.17434)


# 3. Dvoretzky's Theorem
delta = np.float32(1.0)
alpha = lambda n: 1 / n
beta = alpha
for k in range(1, 100001):
    delta = (1 - alpha(k)) * delta + beta(k) * np.random.normal()

print("Delta after Dvoretzky's update:", delta)



