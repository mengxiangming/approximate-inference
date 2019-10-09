import numpy as np

import matplotlib.pyplot as plt

import os

from utils import *

import argparse


save_dir = './results'


def main():
    parser = argparse.ArgumentParser(description='main')

    # signal parameters
    parser.add_argument('--method', type=str, default='Gauss',
                        help='Model name: Gauss,EP,ADF')

    parser.add_argument('--N', type=int, default=50, metavar='N',
                        help='number of observations (default: 4)')

    parser.add_argument('--Iter', type=int, default=2, metavar='N',
                        help='number of iterations of EP (default: 2)')

    parser.add_argument('--save', action='store_true', default=False, help='Enable results saving')



    args = parser.parse_args()

    # problem setting
    prior_mean = 0
    prior_var = 100

    x_true = 5

    shift = 10

    mix1_var = 1
    mix2_var = 5

    NumObservations = args.N


    y = mix_data_generate(x_true, shift, mix1_var, mix2_var, NumObservations)



    # compute the true posterior distribution of x

    post_mean_true,post_var_true,Post_approx_true = true_posterior(y,prior_mean,prior_var, mix1_var, mix2_var, shift)

    legend_name = 'True Posterior, N=' + str(NumObservations)

    # compute the approximate posterior distribution of x
    if args.method == 'Gauss':
        post_mean_gauss,post_var_gauss,Post_approx_method = naivegauss_approximate(y, prior_mean, prior_var, mix1_var, mix2_var, shift)
        legend_name_compare = 'Naive Gauss, N=' + str(NumObservations)

    elif args.method == 'ADF':
        post_mean_adf,post_var_adf,Post_approx_method = adf_approximate(y, prior_mean, prior_var, mix1_var, mix2_var, shift)
        legend_name_compare = 'ADF, N=' + str(NumObservations)

    elif args.method == 'EP':
        legend_name_compare = 'EP(Iter='+str(args.Iter)+'), N=' + str(NumObservations)
        post_mean_ep,post_var_ep,Post_approx_method = ep_approximate(y, prior_mean, prior_var, mix1_var, mix2_var, shift, args.Iter)

    else:
        raise ValueError('No such approximate method!')



    plt.figure()

    x_samples = np.linspace(-10, 20, 1e6)

    plt.plot(x_samples,Post_approx_true,'r--',linewidth=4.0,label=legend_name)


    plt.plot(x_samples, Post_approx_method, 'g-', linewidth=2.0, label=legend_name_compare)



    plt.annotate('True value of x',xy=(x_true, 0), xycoords='data', xytext=(0,0),
                fontsize=16,
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))


    plt.legend(loc='upper left')

    plt.yticks([])
    plt.xticks([])


    if args.save:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        plt.savefig(os.path.join(save_dir, 'img_{}.png'.format(args.method)), dpi=300)


    plt.show()

if __name__ == '__main__':
    main()
    










