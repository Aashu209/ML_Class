import numpy as np

def gradient_search(f, grad_f, x0, lr=0.01, iterations=1000):
    x = np.array(x0, dtype=float)
    for _ in range(iterations):
        x = x - lr * grad_f(x)
    return x

def f(x):
    return np.sum(x**2)

def grad_f(x):
    return 2 * x

x_min = gradient_search(f, grad_f, [5, -3])
print(x_min)

