import numpy as np
import matplotlib.pyplot as plt
from bvt import (
    train_data, get_thetas, get_w, get_hypothesis_1, get_hypothesis_2,
    out_of_sample_error, run_experiment, bias_square, variances, plot_true_target_function_x_y_h1_h2
)

def notebook_tests():
    x_train, y_train  = train_data() 
    print(x_train, y_train)
    assert len(x_train) == 2
    assert len(y_train) == 2
    np.testing.assert_array_equal(np.sin(x_train), y_train)
    for i in range(1000):
        x_tmp, _ = train_data()
        assert x_tmp.min() >= 0.0
        assert x_tmp.max() <= 2*np.pi

    print("train_data test passed!")

    thetas = get_thetas(x_train, y_train)
    w = get_w(x_train, y_train)
    print(thetas[0], thetas[1])
    print(w)   

    x_train_temp = np.array([0, 1])
    y_train_temp = np.array([np.sin(x_i) for x_i in x_train_temp])
    thetas_test = get_thetas(x_train_temp, y_train_temp)
    w_test = get_w(x_train_temp, y_train_temp)

    np.testing.assert_almost_equal(thetas_test[0], 0.0)
    np.testing.assert_almost_equal(thetas_test[1], 0.8414709848078965)
    np.testing.assert_almost_equal(w_test, 0.42073549240394825)

    print("get_thetas and get_w tests passed!")

    # we want to compute numerically the expectation w.r.t. x
    # p(x) is const. in the intervall [0, 2pi]
    x_grid = np.linspace(0, 2*np.pi, 100)
    y_grid = np.sin(x_grid)

    # If your implementation is correct, these tests should not throw an exception

    h1_test = get_hypothesis_1(thetas_test)
    h2_test = get_hypothesis_2(w_test)
    np.testing.assert_almost_equal(h1_test(x_grid)[10], 0.5340523361780719)
    np.testing.assert_almost_equal(h2_test(x_grid)[10], 0.42073549240394825) 

    x_train, y_train  = train_data() 
    hetas = get_thetas(x_train, y_train)
    w = get_w(x_train, y_train)
    plot_true_target_function_x_y_h1_h2(x_train, y_train, get_hypothesis_1(thetas), get_hypothesis_2(w))

    # If your implementation is correct, these tests should not throw an exception

    e_out_h1_test = out_of_sample_error(h1_test(x_grid), y_grid)
    np.testing.assert_almost_equal(e_out_h1_test, 11.52548591)

    # Note: The data for the test have a high out of sample error 11.525
    plt.plot(x_train_temp,y_train_temp, "r*")
    plt.plot(x_grid, h1_test(x_grid), "g-")
    plt.plot(x_grid, y_grid, "b-")

    x_grid.shape
    num_training_data = 10000
    xs, ys, t0s, t1s, ws, e_out_h1s, e_out_h2s = run_experiment(num_training_data)

    t0_avg = t0s.mean()
    t1_avg = t1s.mean()
    thetas_avg = [t0_avg, t1_avg]
    w_avg = ws.mean()
    h1_avg = get_hypothesis_1(thetas_avg)
    h2_avg = get_hypothesis_2(w_avg)
    print(thetas_avg)

    plot_true_target_function_x_y_h1_h2([], [], h1_avg, h2_avg)

    expectation_Eout_1 = e_out_h1s.mean()
    print ("expectation of E_out of model 1:", expectation_Eout_1)

    expectation_Eout_2 = e_out_h2s.mean()
    print ("expectation of E_out of model 2:", expectation_Eout_2)

    bias_1 = bias_square(y_grid,  h1_avg(x_grid))
    print ("Bias of model 1:", bias_1)

    bias_2 = bias_square(y_grid,  h2_avg(x_grid))
    print ("Bias of model 2:", bias_2)

    var_hypothesis_set_1 = variances(get_hypothesis_1, 
                 get_thetas, 
                 xs, ys, 
                 x_grid, 
                 h1_avg(x_grid))
    print(var_hypothesis_set_1)

    var_hypothesis_set_2 = variances(get_hypothesis_2, 
                 get_w, 
                 xs, ys, 
                 x_grid, 
                 h2_avg(x_grid))
    print(var_hypothesis_set_2)

    print("model 1: E_out ≈ bias^2 + variance:  %f ≈ %f + %f" % (expectation_Eout_1, bias_1, var_hypothesis_set_1))
    print("model 2: E_out ≈ bias^2 + variance:  %f ≈ %f + %f" % (expectation_Eout_2, bias_2, var_hypothesis_set_2))

notebook_tests()