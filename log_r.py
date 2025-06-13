# External Modules
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline only needed in Jupyter notebooks

# --- Data Generation

# class 0:
# covariance matrix and mean
cov0 = np.array([[5,-4],[-4,4]])
mean0 = np.array([2.,3])
# number of data points
m0 = 1000

# class 1
# covariance matrix
cov1 = np.array([[5,-3],[-3,3]])
mean1 = np.array([1.,1])
# number of data points
m1 = 1000

# generate m gaussian distributed data points with
# mean and cov.
r0 = np.random.multivariate_normal(mean0, cov0, m0)
r1 = np.random.multivariate_normal(mean1, cov1, m1)

# --- Plotting the Data
plt.scatter(r0[...,0], r0[...,1], c='b', marker='*', label="class 0")
plt.scatter(r1[...,0], r1[...,1], c='r', marker='.', label="class 1")
plt.xlabel("x0")
plt.ylabel("x1")
plt.legend()
plt.show()

X = np.concatenate((r0,r1))
y = np.ones(len(r0)+len(r1))
y[:len(r0),] = 0

# --- Logistic Function
def logistic_function(x):
    return 1 / (1 + np.exp(-x))

# Compute logistic function values
x1 = np.linspace(-10, 10, 100)
y1 = logistic_function(x1)

# Plotting
plt.plot(x1, y1, label='logistic function')
plt.title('Logistic (Sigmoid) Function')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.grid(True)
plt.legend()
plt.show()

# --- Logistic Hypothesis
def logistic_hypothesis(theta):
    return lambda X: logistic_function(
        np.hstack([np.ones((X.shape[0], 1)), X]) @ theta
    )

# Implementation test
#theta = np.array([1.,2.,3.])
#h = logistic_hypothesis(theta)
#print(h(X))

# --- Cross Entropy Loss
def cross_entropy_costs(h, X, y):
    def costs(theta):
        predictions = h(theta)(X)
        eps = 1e-15
        predictions = np.clip(predictions, eps, 1 - eps)
        cost_per_example = - (y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return cost_per_example
    return costs

# Implementation test
theta = np.array([1.,2.,3.])
costs = cross_entropy_costs(logistic_hypothesis, X, y)
print(costs(theta))

# --- Loss Function
def mean_cross_entropy_costs(X, y, hypothesis, cost_func, lambda_reg=0.1):
    def J(theta):
        costs = cost_func(hypothesis, X, y)(theta)
        mean_cost = np.mean(costs)
        reg_term = (lambda_reg / (2 * len(y))) * np.sum(theta[1:] ** 2)
        return mean_cost + reg_term
    return J

# --- Gradient Descent
def compute_new_theta(X, y, theta, learning_rate, hypothesis, lambda_reg=0.1):
    m = len(y)
    h = hypothesis(theta)(X)
    X_prime = np.hstack((np.ones((m, 1)), X))
    error = h - y
    grad = (1 / m) * np.dot(X_prime.T, error)
    reg = (lambda_reg / m) * theta
    reg[0] = 0
    theta_new = theta - learning_rate * (grad + reg)
    return theta_new

def gradient_descent(X, y, theta, learning_rate, num_iters, lambda_reg=0.1):
    J = mean_cross_entropy_costs(X, y, logistic_hypothesis, cross_entropy_costs, lambda_reg)
    history_theta = [theta]
    history_cost = [J(theta)]
    
    for i in range(num_iters):
        theta = compute_new_theta(X, y, theta, learning_rate, logistic_hypothesis, lambda_reg)
        history_theta.append(theta)
        history_cost.append(J(theta))
        
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {J(theta)}")
    
    return history_cost, history_theta

# --- Training and Evaluation
# Choose appropriate values
alpha = 0.1  # learning rate
theta = np.zeros(3)  # initial theta values (includes bias term)
num_iters = 1000  # number of iterations

# Train the model
history_cost, history_theta = gradient_descent(X, y, theta, alpha, num_iters)

# --- Plot Cost Progress
def plot_progress(costs):
    """Plots the costs over the iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('Cost Function J(θ) Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

plot_progress(history_cost)
print("costs before the training:\t ", history_cost[0])
print("costs after the training:\t ", history_cost[-1])

# --- Plot Data and Decision Boundary
# Get the optimized theta values
theta = history_theta[-1]
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', alpha=0.7)

x_values = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
y_values = -(theta[0] + theta[1] * x_values) / theta[2]

plt.plot(x_values, y_values, label='Decision Boundary', color='green')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

# --- Calculate Accuracy
# Make predictions
def accuracy(h, theta, X, y, threshold=0.5):
    probs = h(theta)(X)
    preds = (probs >= threshold).astype(int)
    return np.mean(preds == y)

final_theta = history_theta[-1]
acc = accuracy(logistic_hypothesis, final_theta, X, y)
print(f"Accuracy: {acc:.4f}")

# --- Implementing Regularization
def mean_cross_entropy_costs_with_reg(X, y, hypothesis, cost_func, lambda_reg=0.1):
    m = len(y)
    return lambda theta: np.sum(cost_func(hypothesis, X, y)(theta)) / m + (lambda_reg / (2 * m)) * np.sum(theta[1:]**2)

def compute_new_theta_with_reg(X, y, theta, learning_rate, hypothesis, lambda_reg=0.1):
    m = len(y)
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    h = hypothesis(theta)(X)
    
    # Gradient for theta_0 (bias term) - no regularization
    gradient_0 = (1/m) * X_with_bias.T[0] @ (h - y)
    
    # Gradient for other theta values - with regularization
    gradient_rest = (1/m) * X_with_bias.T[1:] @ (h - y) + (lambda_reg/m) * theta[1:]
    
    # Combine gradients
    gradient = np.zeros_like(theta)
    gradient[0] = gradient_0
    gradient[1:] = gradient_rest
    
    return theta - learning_rate * gradient

def gradient_descent_with_reg(X, y, theta, learning_rate, num_iters, lambda_reg=0.1):
    J = mean_cross_entropy_costs_with_reg(X, y, logistic_hypothesis, cross_entropy_costs, lambda_reg)
    history_theta = [theta]
    history_cost = [J(theta)]
    
    for i in range(num_iters):
        theta = compute_new_theta_with_reg(X, y, theta, learning_rate, logistic_hypothesis, lambda_reg)
        history_theta.append(theta)
        history_cost.append(J(theta))
        
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {J(theta)}")
    
    return history_cost, history_theta
