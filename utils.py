import numpy as np
import math


def mix_data_generate(x_true=5, shift=10, mix1_var=1,mix2_var=5,Num_samples=20):

    y_signal = x_true + np.sqrt(mix1_var) * np.random.randn(Num_samples, 1)
    y_clutter = x_true + shift + np.sqrt(mix2_var) * np.random.randn(Num_samples, 1)

    mixture_coeff = np.where(np.random.rand(Num_samples, 1) > 0.5, np.ones_like(y_signal), np.zeros_like(y_signal))

    Observations = mixture_coeff * y_signal + (1 - mixture_coeff) * y_clutter

    return Observations


def Gauss_density(observation=0,mean=0,var=1):

    return 1/np.sqrt(2*math.pi*var)*np.exp(-(observation-mean)**2/2/var)

def Gauss_density_log(observation=0,mean=0,var=1):

    return -(observation-mean)**2/2/var - 1/2*np.log(2*math.pi*var)


def true_posterior(y,prior_mean=0,prior_var=100, mix1_var = 1,mix2_var=5, shift=10):
    x_samples = np.linspace(-10, 20, 1e6) # for numeraical integration. in fact for this toy problem, closed solution exists.

    #Posterior = Gauss_density(x_samples, prior_mean, prior_var)
    post_density_log = Gauss_density_log(x_samples, prior_mean, prior_var)  # prior factor

    for yi in y:
        # transform to log domain to avoid numerical issue
        post_density_log = post_density_log + np.log(
                    0.5 * Gauss_density(yi, x_samples, mix1_var) + 0.5 * Gauss_density(yi, x_samples + shift, mix2_var))

    post_density_log = post_density_log - max(post_density_log)

    Posterior = np.exp(post_density_log)
    NormFactor = np.sum(Posterior)

    Posterior = Posterior / NormFactor

    post_mean = np.sum(Posterior * x_samples)
    post_var = np.sum(Posterior * ((x_samples - post_mean) ** 2))


    return post_mean,post_var, Posterior



# function of the naive Gaussian approximation of Belief Propagation (BP)
def naivegauss_approximate(y,prior_mean=0,prior_var=100, mix1_var = 1,mix2_var=5, shift=10):
    x_samples = np.linspace(-10, 20, 1e6)
    post_density_app_log = Gauss_density_log(x_samples, prior_mean, prior_var)  # prior factor
    for yi in y:
        temp_mean = 0.5*yi + 0.5*(yi-shift)
        temp_var = 0.5**2*mix1_var + 0.5**2*mix2_var
        #post_density_app = post_density_app * Gauss_density(x_samples, temp_mean, temp_var)  # prior factor
        post_density_app_log = post_density_app_log + Gauss_density_log(x_samples, temp_mean, temp_var)  # transform to log domain to avoid numerical issue

    post_density_app_log = post_density_app_log - max(post_density_app_log)

    post_density_app = np.exp(post_density_app_log)
    post_density_app = post_density_app/np.sum(post_density_app)

    post_mean = np.sum(post_density_app * x_samples)
    post_var = np.sum(post_density_app * ((x_samples - post_mean) ** 2))



    return post_mean, post_var, post_density_app


# function of the assummed density filtering (ADF)
def adf_approximate(y,prior_mean=0,prior_var=100, mix1_var = 1,mix2_var=5, shift=10):

    x_samples = np.linspace(-10, 20, 1e6)  # for numerical compuation

    post_mean = prior_mean
    post_var = prior_var

    for yi in y:
        Posterior_temp = Gauss_density(x_samples,post_mean,post_var) * (0.5*Gauss_density(yi,x_samples,mix1_var) + 0.5*Gauss_density(yi,x_samples+shift,mix2_var))

        post_mean = np.sum(Posterior_temp*x_samples)/np.sum(Posterior_temp)
        post_var = np.sum(Posterior_temp * ((x_samples-post_mean)**2)) / np.sum(Posterior_temp)


    post_density_app = Gauss_density(x_samples, post_mean, post_var)
    post_density_app = post_density_app/np.sum(post_density_app)

    return post_mean, post_var, post_density_app


# function of expectationg propagation (EP)
def ep_approximate(y,prior_mean=0,prior_var=100, mix1_var = 1,mix2_var=5, shift=10,Num_Iter=2):
    # Initialization of the messages form observation factor to variable factor
    mean_factor2variable = np.zeros_like(y)
    var_factor2variable = 1e16*np.ones_like(y)

    x_samples = np.linspace(-10, 20, 1e6) # for numerical compuation

    if len(y) > 1:
        for iter in range(Num_Iter):
            for n in range(len(y)):
                post_var = 1 / (np.sum(1 / var_factor2variable) + 1 / prior_var)
                post_mean = post_var * (np.sum(mean_factor2variable / var_factor2variable) + prior_mean / prior_var)
                # first compute the context mean and variance of the cavity distribution
                var_context = post_var*var_factor2variable[n]/(var_factor2variable[n]-post_var)
                mean_context = (post_mean*var_factor2variable[n] - mean_factor2variable[n]*post_var)/(var_factor2variable[n]-post_var)

                # compute the combined post mean and variance using context mean and variance
                # Note that here we use numerical method, in fact, for this toy expeample, there is closed form solution!!

                Posterior = Gauss_density(x_samples,mean_context,var_context) * (0.5 * Gauss_density(y[n], x_samples, mix1_var) + 0.5 * Gauss_density(y[n], x_samples + shift, mix2_var))

                post_mean_n = np.sum(Posterior*x_samples)/np.sum(Posterior)
                post_var_n = np.sum(Posterior * ((x_samples-post_mean_n)**2)) / np.sum(Posterior)


                # compute the extrinsic mean and var of the Gaussian approximation for factor n
                var_ext = post_var_n*var_context/(var_context - post_var_n)
                mean_ext = (post_mean_n*var_context -  mean_context*post_var_n)/(var_context - post_var_n)

                var_ext = np.where(var_ext<0, 1e10, var_ext)

                var_factor2variable[n] = np.maximum(var_ext, 1e-20) # to avoid zero
                mean_factor2variable[n] = mean_ext

        # after convergence or a maximum number of iterations, return the posterior approx.
        post_var = 1 / (np.sum(1 / var_factor2variable) + 1 / prior_var)
        post_mean = post_var * (np.sum(mean_factor2variable / var_factor2variable) + prior_mean / prior_var)

    elif len(y) == 1: # no need to iterate and the same as ADF
        mean_context = prior_mean
        var_context = prior_var

        Posterior = Gauss_density(x_samples, mean_context, var_context) * (
                0.5 * Gauss_density(y, x_samples, mix1_var) + 0.5 * Gauss_density(y, x_samples + shift,
                                                                                     mix2_var))

        post_mean = np.sum(Posterior * x_samples) / np.sum(Posterior)
        post_var = np.sum(Posterior * ((x_samples - post_mean) ** 2)) / np.sum(Posterior)

    post_density_app = Gauss_density(x_samples, post_mean, post_var)
    post_density_app = post_density_app / np.sum(post_density_app)

    return post_mean, post_var,post_density_app
