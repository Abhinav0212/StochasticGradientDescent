import sys
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns

inputDimension = 5
negativeCenter = -(1.0/5)
positiveCenter = (1.0/5)
num_test_samples = 400

def get_Sample_Point(scenario, sigma):
    """Randomly sample a point from the data distribtion.

    Arguments:
    scenario - selects the input space (1:hypercube, 2:multidimension ball)
    sigma    - standard deviation of gaussian

    Returns:
    List [X,y] where
        X - 6 dimensional input point
        y - expected output value
    """
    # Sample y from {-1,+1} with equal probability.
    y = random.sample([-1,1], 1)[0]

    # Sample each dimension of u from corresponding gaussian depending on y.
    if(y==-1):
        u_vector = [random.gauss(negativeCenter, sigma) for i in range(inputDimension)]
    else:
        u_vector = [random.gauss(positiveCenter, sigma) for i in range(inputDimension)]

    # Project the sample point onto the specified input space.
    x_list = project(scenario, u_vector)
    x_list.append(1)

    return [np.array(x_list),y]

def project(scenario, u_vector):
    """Return the Euclidean projection of a vector in the specified input space.

    Arguments:
    scenario - selects the input space (1:hypercube,2:multidimension ball)
    u_vector - input point

    Returns:
    List - projected point
    """
    # Scenario 1: hypercube, for each dimension if the point is outside the
    # specified space, set it to the closest boundary value.
    if scenario==1:
        for i in range(len(u_vector)):
            if u_vector[i]>1:
                u_vector[i]=1
            elif u_vector[i] < -1:
                u_vector[i]=-1

    # Scenario 2: multidimension ball, if a point is outside the specified space,
    # set it to the unit vector along the same direction.
    elif scenario==2:
        norm_u_vector = norm(u_vector)
        if norm_u_vector>1:
            for i in range(len(u_vector)):
                u_vector[i] = u_vector[i]/norm_u_vector
    return u_vector

def norm(vector):
    """Return the norm of the given input vector."""
    sum=0
    for i in range(len(vector)):
        sum+= (vector[i]*vector[i])
    return np.sqrt(sum)

def gen_test_dataset(scenario, sigma):
    """Return a test dataset.

    Arguments:
    scenario - selects the input space (1:hypercube,2:multidimension ball)
    sigma    - standard deviation of gaussian

    Returns:
    List of List [X,y] where
        X - 6 dimensional input point
        y - expected output value
    """
    return [get_Sample_Point(scenario, sigma) for i in range(num_test_samples)]

def binary_loss(w, x, y):
    """Return the Binary classification error for a given sample.

    Arguments:
    w - weight vector
    x - sample input
    y - expected output

    Returns:
    Integer - 0 or 1
    """
    return int(np.sign(np.dot(w, x)) != y)

def logistic_loss(w, x, y):
    """Return the Logistic loss for a given sample.

    Arguments:
    w - weight vector
    x - sample input
    y - expected output

    Returns:
    float
    """
    return np.log(1 + np.exp(-np.dot(w, x)*y))

def logistic_loss_gradient(W, X, y):
    """Return the gradient of the Logistic loss w.r.t the weight vector.

    Arguments:
    w - weight vector
    x - sample input
    y - expected output

    Returns:
    float
    """
    return -1/(1 + np.exp(y*np.dot(W, X))) * y * X

def sgd(num_training_samples, scenario, sigma):
    """Calculate the output predictor(weight vector) using Stochastic Gradient Descent.

    Arguments:
    num_training_samples - number of SGD iterations
    scenario - selects the input space (1:hypercube,2:multidimension ball)
    sigma    - standard deviation of gaussian

    Returns:
    List - weight vector
    """
    # M is the maximum distance between any two points in the given space.
    # For hypercube, M is equal to the diagonal
    # For multidimension ball, M is the value of the diameter
    if scenario==1:
        M = np.sqrt(np.power(2,2)*(inputDimension+1))
    else:
        M = 2

    # rho = max(||x||) which in both scenarios is half of M.
    rho = M/2
    learning_rate = M/(rho*np.sqrt(num_training_samples))

    # Initialize weight vector to zero.
    W = np.zeros((inputDimension+1))
    W_list = [W]

    # Take a step in the direction given by the oracle and store the value
    # of weights obtained at each iteration.
    for i in range(1,num_training_samples):
        X, y = get_Sample_Point(scenario, sigma)
        W = W - learning_rate*logistic_loss_gradient(W, X, y)
        W = project(scenario, W)
        W_list.append(W)

    # Final hypothesis is the average of weights obtained at each iteration.
    W = np.average(np.array(W_list), axis=0)
    return W

def computeLosses(W_hat_list, test_dataset):
    """Compute estimated binary classification error and estimated excess risk.

    Arguments:
    W_hat_list - List of predictors (weight vectors)
    test_dataset - List of test samples

    Returns:
    List - [Excess risk, std of risk, avg binary error, std binary error]
    """
    logistic_loss_estimate_list = []
    binary_loss_estimate_list = []

    # For each output predictor, estimate the average log loss and average binary
    # classification error across the test dataset and store them.
    for W_hat in W_hat_list:
        logistic_loss_list = [logistic_loss(W_hat, test_sample[0], test_sample[1]) for test_sample in test_dataset]
        average_logistic_loss = np.average(logistic_loss_list)
        binary_loss_list = [binary_loss(W_hat, test_sample[0], test_sample[1]) for test_sample in test_dataset]
        average_binary_loss = np.average(binary_loss_list)

        logistic_loss_estimate_list.append(average_logistic_loss)
        binary_loss_estimate_list.append(average_binary_loss)

    # Estimate the minimum, average and standard deviation of the risks.
    min_risk = np.min(logistic_loss_estimate_list)
    avg_risk = np.average(logistic_loss_estimate_list)
    std_risk = np.std(logistic_loss_estimate_list)

    # Estimate the minimum and average  of the binary classification errors.
    avg_binary_err = np.average(binary_loss_estimate_list)
    std_binary_err = np.std(binary_loss_estimate_list)

    return [avg_risk, std_risk, min_risk, (avg_risk-min_risk), avg_binary_err, std_binary_err]

def errorbar(x_data, y_data, error_data, x_label, y_label, label):
    """Plots the error bar

    Arguments:
    x_data - vector containing data on x-axis
    y_data - vector containing averages
    error_data - vector containing standard deviations
    x_label - plot label to be given to x-axis
    y_label - plot label to be given to x-axis
    title - plot title
    label - plot label for markers
    """
    _, ax = plt.subplots()
    (_, caps, _) = ax.errorbar(x_data, y_data, yerr=error_data, ls='none', fmt='o', markersize=8, capsize=5, label=label)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    for i, v in enumerate(y_data):
        ax.text(x_data[i]+10, v, ('%1.4f' % v).rstrip('0').rstrip('.'), fontsize=8)
    ax.legend()

    for cap in caps:
        cap.set_markeredgewidth(1)

def expirement(scenario):
    """Vary number of iterations and standard deviation to study using SGD's performance.

    Arguments:
    scenario - selects the input space (1:hypercube,2:multidimension ball)
    """

    print("\n============ SCENARIO {} ============\n".format(scenario))
    n_list = [50, 100, 500, 1000]
    sigma_list = [0.05, 0.3]

    for sigma in sigma_list:

        test_dataset = gen_test_dataset(scenario, sigma)

        # Store the expected excess risk, expected classification error and their
        # standard deviations for varying values of n.
        excess_risk_list= []
        std_logistic_loss_list = []
        average_binary_loss_list = []
        std_binary_loss_list = []

        for n in n_list:
            # For each n and sigma, run SGD 30 times and store the output predictor.
            W_hat_list = [sgd(n, scenario, sigma) for i in range(30)]
            # Compute estimated binary classification error and estimated excess risk.
            estimated_losses = computeLosses(W_hat_list, test_dataset)

            print("sigma = ", sigma)
            print("n = ",n)
            print("==Logistic Loss==")
            print("Mean : ",estimated_losses[0])
            print("Standard Deviation : ", estimated_losses[1])
            print("Minimum : ", estimated_losses[2])
            print("Excess risk : ", estimated_losses[3])
            print("==Binary Classification Error==")
            print("Mean : ",estimated_losses[4])
            print("Standard Deviation : ", estimated_losses[5])
            print("\n")

            excess_risk_list.append(estimated_losses[3])
            std_logistic_loss_list.append(estimated_losses[1])
            average_binary_loss_list.append(estimated_losses[4])
            std_binary_loss_list.append(estimated_losses[5])

        # Call the function to create the error bar plot for expected error risk
        errorbar(x_data=n_list
                , y_data=excess_risk_list
                , error_data=std_logistic_loss_list
                , x_label='Number of training examples (n)'
                , y_label='Expected excess risk'
                , label = 'Expected excess risk')
        plt.savefig('plots/scenario' + str(scenario) + "_sigma" + str(sigma) + "_excess_risk" + '.jpg')

        # Call the function to create the error bar plot for binary classification error
        errorbar(x_data=n_list
                , y_data=average_binary_loss_list
                , error_data=std_binary_loss_list
                , x_label='Number of training examples (n)'
                , y_label='Expected binary classification error'
                , label='Expected binary classification error')
        plt.savefig('plots/scenario' + str(scenario) + "_sigma" + str(sigma)+"_binary_error"+'.jpg')

if __name__ == "__main__":
    sys.stdout = open("results.txt", "w")
    expirement(1)
    expirement(2)
    sys.stdout.close()