import numpy as np

def compute_cost(X, y, m, c):
    n = len(y)
    predictions = m * X + c
    cost = (1 / (2 * n)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, m, c, learning_rate, epochs):
    n = len(y)
    for _ in range(epochs):
        predictions = m * X + c
        # Derivatives of the cost function
        dm = (1 / n) * np.sum((predictions - y) * X)
        dc = (1 / n) * np.sum(predictions - y)

        # Update parameters
        m -= learning_rate * dm
        c -= learning_rate * dc
    return m, c

# Sample data: Study hours vs Exam scores
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([35, 50, 65, 80, 95], dtype=float)

# Initialize parameters
m_curr = 0.0
c_curr = 0.0
L = 0.01
epochs = 1000

# Train the model
m_optimized, c_optimized = gradient_descent(X, y, m_curr, c_curr, L, epochs)
print(f"Optimized Slope (m): {m_optimized:.4f}")
print(f"Optimized Intercept (c): {c_optimized:.4f}")

# Make a prediction for 6 hours of study
pred = m_optimized * 6 + c_optimized
print(f"Predicted score for 6 hours of study: {pred:.2f}")