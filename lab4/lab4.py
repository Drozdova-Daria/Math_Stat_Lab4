from distribution import *
from scipy.stats import norm, cauchy, laplace, poisson, uniform, cumfreq
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.special import factorial


def choose_method(distribution):
    if distribution == Distribution.NORMAL:
        return norm
    elif distribution == Distribution.CAUCHY:
        return cauchy
    elif distribution == Distribution.LAPLACE:
        return laplace
    elif distribution == Distribution.POISSON:
        return poisson
    elif distribution == Distribution.UNIFORM:
        return uniform
    else:
        return None


def selection(mu, sigma, size, distribution):
    if distribution == Distribution.NORMAL:
        return norm.rvs(mu, sigma, size)
    elif distribution == Distribution.CAUCHY:
        return cauchy.rvs(mu, sigma, size)
    elif distribution == Distribution.LAPLACE:
        return laplace.rvs(mu, sigma, size)
    elif distribution == Distribution.POISSON:
        return poisson.rvs(mu, size=size)
    elif distribution == Distribution.UNIFORM:
        return uniform.rvs(mu, sigma, size)
    else:
        return None


def distribution_function(x, mu, sigma, distribution):
    if distribution == Distribution.NORMAL:
        return norm.cdf(x, mu, sigma)
    elif distribution == Distribution.CAUCHY:
        return cauchy.cdf(x, mu, sigma)
    elif distribution == Distribution.LAPLACE:
        return laplace.cdf(x, mu, sigma)
    elif distribution == Distribution.POISSON:
        return poisson.cdf(x, mu, sigma)
    elif distribution == Distribution.UNIFORM:
        return uniform.cdf(x, mu, sigma)
    else:
        return None


def density(x, a, b, distribution):
    if distribution == Distribution.NORMAL:
        mu = a
        sigma = np.sqrt(b)
        f = lambda x, sigma, mu: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        return f(x, sigma, mu)
    elif distribution == Distribution.CAUCHY:
        return cauchy.pdf(x, a, b)
    elif distribution == Distribution.LAPLACE:
        return laplace.pdf(x, a, b)
    elif distribution == Distribution.POISSON:
        f = lambda x, a: np.exp(-a) * np.power(a, x) / factorial(x)
        return f(x, a)
    elif distribution == Distribution.UNIFORM:
        return uniform.pdf(x, a, b)
    else:
        return None


def build_empirical_function(a, b, size, distribution, color, interval):
    s = selection(a, b, size, distribution)
    x = np.linspace(interval[0], interval[1], 1000)
    ecdf = sm.distributions.ECDF(s)
    y = ecdf(x)

    distr_f = distribution_function(x, a, b, distribution)
    with sns.plotting_context(font_scale=1.5), sns.axes_style('whitegrid'):
        plt.figure(figsize=(12, 7))
        plt.title('Empirical function: ' + Distribution.in_str(distribution) + ' ' + str(size))
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.step(x, y, label='cdf')
        plt.plot(x, distr_f, linewidth=1, color=color, label='F(x)')
        plt.legend()
        plt.savefig('emp' + Distribution.in_str(distribution) + str(size) + '.png')
        #plt.show()


def build_density(a, b, size, distribution, color, bw, interval):
    s = selection(a, b, size, distribution)
    x = np.linspace(interval[0], interval[1], 1000)
    with sns.plotting_context(font_scale=1.5), sns.axes_style('whitegrid'):
        plt.figure(figsize=(12, 7))
        plt.title('Kernel Density Estimation: ' + Distribution.in_str(distribution) + ' ' + str(size) + ' bw=' + str(bw))
        plt.xlabel('x')
        plt.ylabel('f(x)')
        sns.kdeplot(data=s, bw_method='silverman', bw_adjust=bw, color=color, clip=interval, legend=True, label='kde')
        if Distribution.distribution_type(distribution) == DistributionType.DISCRETE:
            n = np.arange(poisson.ppf(0.001, a), poisson.ppf(0.999, a))
            plt.plot(n, poisson.pmf(n, a), label='f(x)')
        else:
            sns.distplot(s, kde=False, fit=choose_method(distribution), rug=False, hist=False,  label='f(x)')
        plt.xlim(list(interval))
        plt.legend()
        plt.savefig('den' + Distribution.in_str(distribution) + str(size) + str(bw) + '.png')



size = [20, 60, 100]
a_parameters = [0, 0, 0, 10, -np.sqrt(3)]
b_parameters = [1, 1, np.sqrt(2), 0, np.sqrt(3)]
distributions = [Distribution.NORMAL, Distribution.CAUCHY, Distribution.LAPLACE, Distribution.POISSON, Distribution.UNIFORM]
bw = [0.5, 1, 2]
interval_continuous = (-4, 4)
interval_discrete = (6, 14)
colors = ['blue', 'red', 'green', 'pink', 'gray']

for a, b, distribution, color in zip(a_parameters, b_parameters, distributions, colors):
    if Distribution.distribution_type(distribution) == DistributionType.CONTINUOUS:
        interval = interval_continuous
    else:
        interval = interval_discrete
    for n in size:
        build_empirical_function(a, b, n, distribution, color, interval)
        for b in bw:
            build_density(a, b, n, distribution, color, b, interval)
