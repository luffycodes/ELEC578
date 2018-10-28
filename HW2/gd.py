import numpy as np

prod = 1
c = 1
for k in range(1, 10000, 1):
    c = c * 6
    prod = prod * 0.8
    if prod < 0.01:
        print(k)
        break
print(prod)

sigma = 0
c = 0
for k in range(1, 100000000, 1):
    c = c + 1
    sigma = sigma + (1 / c)
    if sigma > 7.5 - 0.01*7.5:
        print(k)
        break
print(sigma)

prod = 1
c = 1
for k in range(1, 100000, 1):
    c = c * 6
    prod = prod * (1 - 2/c)
    if prod < 0.01:
        print(k)
        break
print(prod)

prod = 1
c = 0
for k in range(1, 100000000, 1):
    c = c + 1
    prod = prod * (1 - 2/(4 * c))
    if prod < 0.01:
        print(k)
        break
print(prod)
