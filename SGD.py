import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

inputDimension = 5
negativeCenter = -(1.0/5)
positiveCenter = (1.0/5)
num_test_samples = 400

def get_Sample_Point(scenario, sigma):
    y = random.sample([-1,1], 1)[0]
    if(y==-1):
        u_vector = [random.gauss(negativeCenter, sigma) for i in range(inputDimension)]
    else:
        u_vector = [random.gauss(negativeCenter, sigma) for i in range(inputDimension)]
    x_list = project(scenario, u_vector)
    x_list.append(1)

    return np.array(x_list),y

def project(scenario, u_vector):
    if scenario==1:
        for i in range(len(u_vector)):
            if u_vector[i]>1:
                u_vector[i]=1
            elif u_vector[i] < -1:
                u_vector[i]=-1
    return u_vector

def gen_test_dataset(scenario, sigma):
    return [get_Sample_Point(scenario, sigma) for i in range(num_test_samples)]

def binary_loss(w, x, y):
    return int(np.sign(np.dot(w, x)) == y)

def logistic_loss(w, x, y):
    return np.log(1 + np.exp(-np.dot(w, x)*y))

def logistic_loss_gradient(W, X, y):
    # print(y)
    # print(W.shape)
    # print(X.shape)
    # print(np.dot(W, X))
    # print(np.exp(y*np.dot(W, X)))
    # print(-1/(1 + np.exp(y*np.dot(W, X))))
    return -1/(1 + np.exp(y*np.dot(W, X))) * y * X

def sgd(num_training_samples, scenario, sigma):

    if scenario==1:
        M = np.sqrt(np.power(2,2)*(inputDimension+1))
    else:
        M = 2
    rho = M/2
    learning_rate = M/(rho*np.sqrt(num_training_samples))
    W = np.zeros((inputDimension+1))

    if scenario==1:
        M = np.sqrt(np.power(2,2)*(inputDimension+1))
    else:
        M = 2
    rho = M/2
    learning_rate = M/(rho*np.sqrt(num_training_samples))
    W = np.zeros((inputDimension+1))

    W_list = [W]

    for i in range(1,num_training_samples):
        X, y = get_Sample_Point(scenario, sigma)
        W = W - learning_rate*logistic_loss_gradient(W, X, y)
        W_projected = project(scenario, W)
        W_list.append(W_projected)
        W = W_projected

    W = np.average(np.array(W_list), axis=0)
    return W

def expirement(scenario):

    n_list = [50, 100, 500, 1000]
    sigma_list = [0.05, 3]

    for sigma in sigma_list:

        test_dataset = gen_test_dataset(scenario, sigma)

        excess_risk_list= []
        std_logistic_loss_list = []
        average_binary_loss_list = []
        std_binary_loss_list = []

        for n in n_list:

            W_hat_list = [sgd(n, scenario, sigma) for i in range(30)]

            logistic_loss_estimate_list = []
            binary_loss_estimate_list = []

            for W_hat in W_hat_list:
                logistic_loss_list = [logistic_loss(W_hat, test_sample[0], test_sample[1]) for test_sample in test_dataset]
                average_logistic_loss = np.average(logistic_loss_list)
                binary_loss_list = [binary_loss(W_hat, test_sample[0], test_sample[1]) for test_sample in test_dataset]
                average_binary_loss = np.average(binary_loss_list)

                logistic_loss_estimate_list.append(average_logistic_loss)
                binary_loss_estimate_list.append(average_binary_loss)

            min_logistic_loss = np.min(logistic_loss_estimate_list)
            avg_logistic_loss = np.average(logistic_loss_estimate_list)
            std_logistic_loss = np.std(logistic_loss_estimate_list)

            avg_binary_loss = np.average(binary_loss_estimate_list)
            std_binary_loss = np.std(binary_loss_estimate_list)

            excess_risk_list.append(avg_logistic_loss-min_logistic_loss)
            std_logistic_loss_list.append(std_logistic_loss)
            average_binary_loss_list.append(avg_binary_loss)
            std_binary_loss_list.append(std_binary_loss)

        plt.errorbar(n_list, excess_risk_list, std_logistic_loss_list, linestyle='None', marker='^')
        plt.show()
        print(std_binary_loss_list)
        plt.errorbar(n_list, average_binary_loss_list, std_binary_loss_list, linestyle='None', marker='^')
        plt.show()

expirement(1)