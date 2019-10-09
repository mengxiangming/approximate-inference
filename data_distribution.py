import numpy as np

import matplotlib.pyplot as plt

import math

from utils import *


def main():

    # generate data samples

    x_true = 5

    shift = 10

    mix1_var = 1
    mix2_var = 5

    Num_samples = 100

    y = mix_data_generate(x_true, shift, mix1_var, mix2_var, Num_samples)


    # plot the sampling data y
    plt.plot(y,0.05*np.ones_like(y),'+')


    # plot the density
    x = np.linspace(0,20,1000)

    y1 = 1/np.sqrt(2*math.pi*1)*np.exp(-(x-x_true)**2/2)
    y2 = 1/np.sqrt(2*math.pi*5)*np.exp(-(x-x_true-shift)**2/5)
    y_mixture = 0.5*Gauss_density(x,x_true,1)+0.5*Gauss_density(x,x_true+shift,5)

    plt.plot(x,y1,'g--',linewidth=3.0,label='component 1')
    plt.plot(x,y2,'b--',linewidth=3.0,label='component 2')
    plt.plot(x,y_mixture,'r',linewidth=3.0,label='mixture ')

    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    main()