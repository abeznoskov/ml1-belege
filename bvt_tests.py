import numpy as np
import matplotlib.pyplot as plt
from bvt import (
    train_data, get_thetas, get_w, get_hypothesis_1, get_hypothesis_2,
    out_of_sample_error, run_experiment, bias_square, variances
)

def test_train_data():
    """Test that train_data generates correct data"""
    for _ in range(10):
        x_train, y_train = train_data()
        
        # Check shape
        assert len(x_train) == 2
        assert len(y_train) == 2
        
        # Check value ranges for x
        assert np.all(x_train >= 0.0)
        assert np.all(x_train <= 2*np.pi)
        
        # Check that y = sin(x)
        np.testing.assert_array_almost_equal(np.sin(x_train), y_train)
    
    print("train_data test passed!")

def test_get_thetas():
    """Test get_thetas with known values"""
    # Test case 1: Simple line
    x = np.array([0, 1])
    y = np.array([0, 1])
    thetas = get_thetas(x, y)
    assert len(thetas) == 2
    np.testing.assert_almost_equal(thetas[0], 0.0)
    np.testing.assert_almost_equal(thetas[1], 1.0)
    
    # Test case 2: Using sin values
    x = np.array([0, 1])
    y = np.array([np.sin(0), np.sin(1)])
    thetas = get_thetas(x, y)
    assert len(thetas) == 2
    np.testing.assert_almost_equal(thetas[0], 0.0)
    np.testing.assert_almost_equal(thetas[1], 0.8414709848078965)
    
    # Test case 3: Vertical line (x1 = x2)
    x = np.array([1, 1])
    y = np.array([0, 2])
    thetas = get_thetas(x, y)
    assert len(thetas) == 2
    np.testing.assert_almost_equal(thetas[0], 1.0)
    np.testing.assert_almost_equal(thetas[1], 0.0)
    
    print("get_thetas test passed!")

def test_get_w():
    """Test get_w with known values"""
    # Test case 1: Simple average
    x = np.array([0, 1])  # x is not used by get_w
    y = np.array([1, 3])
    w = get_w(x, y)
    np.testing.assert_almost_equal(w, 2.0)
    
    # Test case 2: Using sin values
    x = np.array([0, 1])
    y = np.array([np.sin(0), np.sin(1)])
    w = get_w(x, y)
    np.testing.assert_almost_equal(w, 0.42073549240394825)
    
    print("get_w test passed!")

def test_get_hypothesis_1():
    """Test the linear hypothesis function"""
    thetas = [2, 3]  # θ0 = 2, θ1 = 3
    h1 = get_hypothesis_1(thetas)
    
    # Test for single value
    np.testing.assert_almost_equal(h1(1), 5)
    
    # Test for array
    x = np.array([0, 1, 2])
    expected = np.array([2, 5, 8])
    np.testing.assert_array_almost_equal(h1(x), expected)
    
    print("get_hypothesis_1 test passed!")

def test_get_hypothesis_2():
    """Test the constant hypothesis function"""
    w = 4
    h2 = get_hypothesis_2(w)
    
    # Test for single value
    np.testing.assert_almost_equal(h2(1), 4)
    
    # Test for array
    x = np.array([0, 1, 2])
    expected = np.array([4, 4, 4])
    np.testing.assert_array_almost_equal(h2(x), expected)
    
    print("get_hypothesis_2 test passed!")

def test_out_of_sample_error():
    """Test the out of sample error calculation"""
    y_preds = np.array([1, 2, 3])
    y = np.array([2, 3, 4])
    # MSE should be ((1-2)² + (2-3)² + (3-4)²)/3 = (1 + 1 + 1)/3 = 1
    error = out_of_sample_error(y_preds, y)
    np.testing.assert_almost_equal(error, 1.0)
    
    print("out_of_sample_error test passed!")

def test_bias_square():
    """Test the bias_square calculation"""
    # Test with simple arrays
    y_true = np.array([1, 2, 3])
    y_avg = np.array([2, 3, 4])
    # Bias² should be ((2-1)² + (3-2)² + (4-3)²)/3 = (1 + 1 + 1)/3 = 1
    bias_sq = bias_square(y_true, y_avg)
    np.testing.assert_almost_equal(bias_sq, 1.0)
    
    print("bias_square test passed!")

def test_run_experiment():
    """Test that run_experiment runs properly"""
    m = 10  # Small number for testing
    xs, ys, t0s, t1s, ws, e_out_h1s, e_out_h2s = run_experiment(m)
    
    # Check shapes
    assert xs.shape == (m, 2)
    assert ys.shape == (m, 2)
    assert t0s.shape == (m,)
    assert t1s.shape == (m,)
    assert ws.shape == (m,)
    assert e_out_h1s.shape == (m,)
    assert e_out_h2s.shape == (m,)
    
    # Check that y values are sine of x values
    for i in range(m):
        np.testing.assert_array_almost_equal(np.sin(xs[i]), ys[i])
    
    print("run_experiment test passed!")

def test_variances():
    """Test the variance calculation"""
    # Create a simple test scenario
    m = 3  # Number of experiments
    xs = np.array([
        [0, 1],   # First experiment
        [2, 3],   # Second experiment
        [4, 5]    # Third experiment
    ])
    ys = np.array([
        [0, 1],   # y values for first experiment
        [1, 0],   # y values for second experiment
        [0, 1]    # y values for third experiment
    ])
    
    # Define simple parameter and hypothesis functions for testing
    def test_param_func(x, y):
        return 1.0  # Just return constant parameter
    
    def test_hypothesis_func(param):
        def h(x):
            return np.ones_like(x) * param
        return h
    
    x_grid = np.array([0, 1, 2])
    y_avg = np.array([1, 1, 1])
    
    # Expected variance: all hypotheses return constant 1, which equals the average hypothesis
    # So variance should be 0
    var = variances(test_hypothesis_func, test_param_func, xs, ys, x_grid, y_avg)
    np.testing.assert_almost_equal(var, 0.0)
    
    print("variances test passed!")

if __name__ == "__main__":
    test_train_data()
    test_get_thetas()
    test_get_w()
    test_get_hypothesis_1()
    test_get_hypothesis_2()
    test_out_of_sample_error()
    test_bias_square()
    test_run_experiment()
    test_variances()
    print("All tests passed!")