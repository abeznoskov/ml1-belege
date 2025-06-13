import numpy as np
import matplotlib.pyplot as plt
import hashlib

def round_and_hash(value, precision=4, dtype=np.float32):
    """ 
    Function to round and hash a scalar or numpy array of scalars.
    Used to compare results with true solutions without spoiling the solution.
    """
    rounded = np.array([value], dtype=dtype).round(decimals=precision)
    hashed = hashlib.md5(rounded).hexdigest()
    return hashed

def train_data():
    """Generate two random training examples from uniform distribution [0,2π]"""
    x = np.random.uniform(0, 2 * np.pi, 2)
    y = np.sin(x)
    return x, y

def get_thetas(x, y):
    """Calculate θ0 and θ1 for linear hypothesis h1(x) = θ0 + θ1*x"""
    # For a line through two points (x1,y1) and (x2,y2):
    # θ1 = (y2-y1)/(x2-x1)
    # θ0 = y1 - θ1*x1
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    
    # Handle the case where x1 == x2 to avoid division by zero
    if x1 == x2:
        theta1 = 0
        theta0 = (y1 + y2) / 2
    else:
        theta1 = (y2 - y1) / (x2 - x1)
        theta0 = y1 - theta1 * x1
        
    return [theta0, theta1]

def get_w(x, y):
    """Calculate w for constant hypothesis h2(x) = w"""
    # w is just the average of the y values
    return np.mean(y)

def get_hypothesis_1(thetas):
    """Return the linear hypothesis function h1(x) = θ0 + θ1*x"""
    def h1(x):
        return thetas[0] + thetas[1] * x
    return h1

def get_hypothesis_2(w):
    """Return the constant hypothesis function h2(x) = w"""
    def h2(x):
        return np.ones_like(x) * w
    return h2

def plot_true_target_function_x_y_h1_h2(x, y, hypothesis1, hypothesis2):
    """Plot the true target function, training examples, and both hypotheses"""
    # Create the grid for plotting
    x_grid = np.linspace(0, 2*np.pi, 100)
    y_grid = np.sin(x_grid)
    
    plt.figure(figsize=(10, 6))
    # Plot the training examples if provided
    if len(x) > 0 and len(y) > 0:
        plt.plot(x, y, 'ro', markersize=8, label='Training examples')
    
    # Plot the true target function sin(x)
    plt.plot(x_grid, y_grid, 'b-', linewidth=2, label='Target function: sin(x)')
    
    # Plot the hypotheses
    plt.plot(x_grid, hypothesis1(x_grid), 'g-', linewidth=2, label='h1(x) = θ0 + θ1*x')
    plt.plot(x_grid, hypothesis2(x_grid), 'm-', linewidth=2, label='h2(x) = w')
    
    plt.title('Sin function approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

def out_of_sample_error(y_preds, y):
    """Calculate the out of sample error using mean squared error"""
    return np.mean((y_preds - y) ** 2)

def run_experiment(m):
    """Run m experiments, each with two training examples"""
    xs = np.ndarray((m, 2))
    ys = np.ndarray((m, 2))
    t0s = np.ndarray(m)
    t1s = np.ndarray(m)
    ws = np.ndarray(m)
    e_out_h1s = np.ndarray(m)
    e_out_h2s = np.ndarray(m)
    
    # Create grid for out-of-sample error calculation
    x_grid = np.linspace(0, 2*np.pi, 100)
    y_grid = np.sin(x_grid)
    
    for i in range(m):
        # Generate training data
        x_train, y_train = train_data()
        xs[i] = x_train
        ys[i] = y_train
        
        # Calculate parameters for both hypotheses
        theta0, theta1 = get_thetas(x_train, y_train)
        t0s[i] = theta0
        t1s[i] = theta1
        
        w = get_w(x_train, y_train)
        ws[i] = w
        
        # Get hypothesis functions
        h1 = get_hypothesis_1([theta0, theta1])
        h2 = get_hypothesis_2(w)
        
        # Calculate out-of-sample errors
        e_out_h1s[i] = out_of_sample_error(h1(x_grid), y_grid)
        e_out_h2s[i] = out_of_sample_error(h2(x_grid), y_grid)
    
    return xs, ys, t0s, t1s, ws, e_out_h1s, e_out_h2s

def bias_square(y_true, y_avg):
    """
    Returns the bias^2 of a hypothesis set for the sin-example.

        Parameters:
                y_true(np.array): The y-values of the target function
                                  at each position on the x_grid
                y_avg(np.array): The y-values of the avg hypothesis 
                                 at each position on the x_grid

        Returns:
                variance (double): Bias^2 of the hypothesis set
    """
    # Bias^2 is the expected squared difference between 
    # average hypothesis and target function
    return np.mean((y_avg - y_true) ** 2)

def variances(hypothesis_func, param_func, xs, ys, x_grid, y_avg):
    '''
    Returns the variance of a hypothesis set for the sin-example.

            Parameters:
                    hypothesis_func (function): The hypothesis function 1 or 2
                    param_func (function): the function to calculate the parameters
                            from the training data, i.e., get_theta or get_w 
                    xs(np.array): 2D-Array with different training data values for x
                                first dimension: differerent training data sets
                                second dimension: data points in a data set
                    ys(np.array): 2D-Array with different training data values for y
                                first dimension: differerent training data sets
                                second dimension: data points in a data set
                    x_grid(np.array): The x-values for calculating the expectation E_x
                    y_avg(np.array): The y-values of the average hypothesis at the 
                                     positions of x_grid

            Returns:
                    variance (double):  Variance of the hypothesis set for 
                                        a type for training data 
                                        (here two examples per training data set)
    '''
    # Number of experiments
    n_experiments = xs.shape[0]
    
    # Calculate the variance as mean squared difference between 
    # individual hypotheses and average hypothesis
    var_sum = 0
    for i in range(n_experiments):
        x_train = xs[i]
        y_train = ys[i]
        
        # Get parameters for current training data
        params = param_func(x_train, y_train)
        
        # Get hypothesis function for current parameters
        h = hypothesis_func(params)
        
        # Calculate hypothesis predictions on the grid
        y_pred = h(x_grid)
        
        # Sum the squared difference between prediction and average hypothesis
        var_sum += np.mean((y_pred - y_avg) ** 2)
    
    # Return average variance
    return var_sum / n_experiments