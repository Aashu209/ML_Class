import numpy as np
import pandas as pd

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs

    def fit(self, X, y):
        self.m, self.n = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(self.n)
        self.b = 0

        for _ in range(self.epochs):
            for idx, x in enumerate(X):
                condition = y_[idx] * (np.dot(x, self.w) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_[idx] * x)
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# ---------------- LOAD CSV ---------------- #

df = pd.read_csv("data.csv")

X = df.iloc[:, :-1].values   # all columns except label
y = df.iloc[:, -1].values   # label column

# ---------------- TRAIN MODEL ---------------- #

model = SVM(lr=0.001, epochs=1000)
model.fit(X, y)

# ---------------- PREDICT ---------------- #

predictions = model.predict(X)

print("Predictions:", predictions)
print("Actual labels:", y)
print("Weights:", model.w)
print("Bias:", model.b)
