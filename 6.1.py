import numpy as np

# This is the code for 6.1
P = np.array([[0.1, 0.2, 0.3, 0.4], [0, 0, 1.0, 0], [0, 0, 0, 1.0], [1.0, 0, 0, 0]])
x_1 = P[0]
n = 100000
periodic = False
output_individual = False
period = 2  # if periodic adjust period here

print(P)
print(x_1)
x_is = {}
x_is[0] = x_1

for i in range(n):
    x_is[i + 1] = x_1 @ P
    x_1 = x_is[i + 1]

if output_individual:
    for key in x_is.keys():
        print(f"x_{key} = {x_is[key]}")


print(x_is[n])

if periodic:

    final_vec = np.zeros(shape=np.shape(x_1))

    for i in range(period):
        final_vec = np.add(final_vec, x_is[n - i])

    final_vec = final_vec * (1 / period)
    print(final_vec)
