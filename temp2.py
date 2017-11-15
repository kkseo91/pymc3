import theano
floatX = theano.config.floatX
import pymc3 as pm
import math
import theano.tensor as T
import sklearn
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
X1,Y1= make_moons(noise=0.01, random_state=0, n_samples=100)
X1 = X1+ 1
X1 = X1.astype(floatX)
Y1 = Y1.astype(floatX)
X2,Y2= make_moons(noise=0.01, random_state=0, n_samples=100)
X2 = X2+2
X2 = X2.astype(floatX)
Y2 = Y2.astype(floatX)
X3,Y3= make_moons(noise=0.01, random_state=0, n_samples=100)
X3 = X3 +3
X3 = X3.astype(floatX)
Y3 = Y3.astype(floatX)
X4,Y4= make_moons(noise=0.01, random_state=0, n_samples=100)
X4 = X4+ 4
X4 = X4.astype(floatX)
Y4 = Y4.astype(floatX)
X5,Y5= make_moons(noise=0.01, random_state=0, n_samples=100)
X5 = X5 +5
X5 = X5.astype(floatX)
Y5 = Y5.astype(floatX)
X6,Y6= make_moons(noise=0.01, random_state=0, n_samples=100)
X6 = X1 + 6
X6 = X6.astype(floatX)
Y6 = Y6.astype(floatX)
basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    alpha1 = pm.Normal('alpha1', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10, shape=2)
    alpha2 = pm.Normal('alpha2', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10, shape=2)
    #sigma1 = pm.Normal('sigma1', mu=0, sd=10, shape=1)
    #sigma2 = pm.Normal('sigma2', mu=0, sd=10, shape=1)
    mu = np.array([0,0,0,0,0,0])
    cov =  np.array( [[1,1/2,1/3,1/4,1/5,1/6], [1/2,1,1/2,1/3,1/4,1/5], [1/3,1/2,1,1/2,1/3,1/4], [1/4,1/3,1/2,1,1/2,1/3], [1/5,1/4,1/3,1/2,1,1/2], [1/6,1/5,1/4,1/3,1/2,1]])
    beta_s1 = pm.MvNormal('beta_s1', mu, cov, shape=(6))
    mu = [0, 0, 0, 0, 0, 0]
    cov2 = np.array([[1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6], [1 / 2, 1, 1 / 2, 1 / 3, 1 / 4, 1 / 5],
                   [1 / 3, 1 / 2, 1, 1 / 2, 1 / 3, 1 / 4], [1 / 4, 1 / 3, 1 / 2, 1, 1 / 2, 1 / 3],
                   [1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 1 / 2], [1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1]])
    beta_s2 = pm.MvNormal('beta_s2', mu, cov2, shape=6)
    gamma1 = pm.Normal('gamma1', mu=0, sd=10)
    gamma2 = pm.Normal('gamma2', mu=0, sd=10)

    mu1_1 = pm.math.tanh(X1[:,0]*(beta_s1[0]+beta1[0]) + X1[:,1]*(beta_s2[0]+ beta1[1]) +alpha1)
    mu2_1 = pm.math.tanh(X1[:,0]*(beta_s1[0]+beta2[0]) + X1[:,1]*(beta_s2[0]+ beta2[1]) +alpha2)
    act_out1 = pm.math.tanh(gamma1*mu1_1 + gamma2*mu2_1)
    out1 = pm.Bernoulli('out1',act_out1,observed= Y1,
                       total_size=Y1.shape[0]  # IMPORTANT for minibatches
                       )


    mu1_2 = pm.math.tanh(X2[:,0]*(beta_s1[1]+beta1[0]) + X2[:,1]*(beta_s2[1]+ beta1[1]) +alpha1)
    mu2_2 = pm.math.tanh(X2[:,0]*(beta_s1[1]+beta2[0]) + X2[:,1]*(beta_s2[1]+ beta2[1]) +alpha2)
    act_out2 = pm.math.tanh(gamma1*mu1_2 + gamma2*mu2_2)
    out2 = pm.Bernoulli('out2',act_out2,observed= Y2,
                       total_size=Y2.shape[0]  # IMPORTANT for minibatches
                       )

    mu1_3 = pm.math.tanh(X3[:, 0] * (beta_s1[2] + beta1[0]) + X3[:, 1] * (beta_s2[2] + beta1[1]) + alpha1)
    mu2_3 = pm.math.tanh(X3[:, 0] * (beta_s1[2] + beta2[0]) + X3[:, 1] * (beta_s2[2] + beta2[1]) + alpha2)
    act_out3 = pm.math.tanh(gamma1 * mu1_3 + gamma2 * mu2_3)
    out3 = pm.Bernoulli('out3', act_out3, observed=Y3,
                        total_size=Y3.shape[0]  # IMPORTANT for minibatches
                        )


    mu1_4 = pm.math.tanh(X4[:, 0] * (beta_s1[3] + beta1[0]) + X4[:, 1] * (beta_s2[3] + beta1[1]) + alpha1)
    mu2_4 = pm.math.tanh(X4[:, 0] * (beta_s1[3] + beta2[0]) + X4[:, 1] * (beta_s2[3] + beta2[1]) + alpha2)
    act_out4 = pm.math.tanh(gamma1 * mu1_4 + gamma2 * mu2_4)
    out4 = pm.Bernoulli('out4', act_out4, observed=Y4,
                        total_size=Y4.shape[0]  # IMPORTANT for minibatches
                        )


    mu1_5 = pm.math.tanh(X5[:, 0] * (beta_s1[4] + beta1[0]) + X5[:, 1] * (beta_s2[4] + beta1[1]) + alpha1)
    mu2_5 = pm.math.tanh(X5[:, 0] * (beta_s1[4] + beta2[0]) + X5[:, 1] * (beta_s2[4] + beta2[1]) + alpha2)
    act_out5 = pm.math.tanh(gamma1 * mu1_5 + gamma2 * mu2_5)
    out5 = pm.Bernoulli('out5', act_out5, observed=Y5,
                        total_size=Y5.shape[0]  # IMPORTANT for minibatches
                        )


    mu1_6 = pm.math.tanh(X6[:, 0] * (beta_s1[5] + beta1[0]) + X6[:, 1] * (beta_s2[5] + beta1[1]) + alpha1)
    mu2_6 = pm.math.tanh(X6[:, 0] * (beta_s1[5] + beta2[0]) + X6[:, 1] * (beta_s2[5] + beta2[1]) + alpha2)
    act_out6 = pm.math.tanh(gamma1 * mu1_6 + gamma2 * mu2_6)
    out6 = pm.Bernoulli('out6', act_out6, observed=Y6,
                        total_size=Y6.shape[0]  # IMPORTANT for minibatches
                        )
    inference = pm.ADVI()
    approx = pm.fit(n=80000, method=inference)
    trace = approx.sample(draws=10000)
    pm.summary(trace)
    _ = pm.traceplot(trace)
    plt.show()
    beta1s= np.mean(trace.beta_s1[0])
    beta2s = np.mean(trace.beta_s2[0])
    beta1 = np.mean(trace.beta1)
    mu1_1 = np.tanh(X1[:,0]*(beta1s+np.mean(trace.beta1[0])) + X1[:,1]*(beta2s+ np.mean(trace.beta1[1]))+ np.mean(trace.alpha1))
    mu2_1 = np.tanh(X1[:,0]*(beta1s+np.mean(trace.beta2[0])) + X1[:,1]*(beta2s+ np.mean(trace.beta2[1])) +np.mean(trace.alpha2))
    pred = np.tanh(np.mean(trace.gamma1)*mu1_1 + np.mean(trace.gamma2)*mu2_1)
    print (pred)
    pred = pred > 0.5
    print (pred)
    fig, ax = plt.subplots()
    ax.scatter(X1[pred==0, 0], X1[pred==0, 1])
    ax.scatter(X1[pred==1, 0], X1[pred==1, 1], color='r')
    sns.despine()
    ax.set(title='Predicted labels in testing set', xlabel='X', ylabel='Y');
    plt.show()