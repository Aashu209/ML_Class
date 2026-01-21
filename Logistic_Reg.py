import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0

        for _ in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            y_hat = self.sigmoid(z)
            dw = (1 / self.m) * np.dot(X.T, (y_hat - y))
            db = (1 / self.m) * np.sum(y_hat - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_hat = self.sigmoid(z)
        return (y_hat >= 0.5).astype(int)


# ---------------- RUN THE MODEL ---------------- #

# Sample dataset
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

y = np.array([0, 0, 1, 1])

# Create model
model = LogisticRegression(lr=0.1, epochs=1000)

# Train
model.fit(X, y)

# Predict
predictions = model.predict(X)

print("Predictions:", predictions)
print("Weights:", model.w)
print("Bias:", model.b)
