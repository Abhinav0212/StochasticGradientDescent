import random
import numpy as np
import math

inputDimension = 5
negativeCenter = -(1.0/5)
positiveCenter = (1.0/5)
num_test_samples = 400

def get_Sample_Point(scenario, sigma):
    y = random.sample([-1,1], 1)
    if(y==-1):
        u_vector = [random.gauss(negativeCenter, sigma) for i in range(inputDimension)]
    else:
        u_vector = [random.gauss(negativeCenter, sigma) for i in range(inputDimension)]
    x_list = project(senario, u_vector)
    x_list.append(1)

    return np.array(x_list),y

def project(scneario, u_vector):
    if scenario==1:
        for i in range(inputDimension):
            if(u_vector[i]>1):
                u_vector[i]=1
            else if(u_vector[i]<-1):
                u_vector[i]=-1
    return u_vector

def gen_test_dataset(scenario, sigma):
    [get_Sample_Point(scenario, sigma) for i in range(num_test_samples)]

def sgd(num_training_samples, scenario, sigma):
