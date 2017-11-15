import theano
floatX = theano.config.floatX
import pymc3 as pm
from scipy import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')

sns.set_style('white')

#Data import

df=pd.read_csv('topsoil.csv', sep=',',header=None)
topsoil = np.array(df.values)
topsoil_id_old = topsoil[:,3]

df=pd.read_csv('adj.csv', sep=',',header=None)
adj = np.array(df.values)

df=pd.read_csv('sediment.csv', sep=',',header=None)
sediment = np.array(df.values)
sediment_id_old = sediment[:,3]

df=pd.read_csv('id_change.csv', sep=',',header=None)
id_change = np.array(df.values)

def new_id(old_id, id_change):
    n = len(old_id)
    new_id = []
    for i in range(n):
        index, = np.where(id_change[:,0] == old_id[i])
        new_id.append(id_change[:,1][index][0])
    return new_id

def distanceM(S1, S2):
    n = len(S1)
    distM =np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distM[i][j] = ((S1[i]-S1[j])**2 + (S2[i]-S2[j])**2)**0.5
    return distM

topsoil_id  = np.array(new_id(topsoil_id_old, id_change))
sediment_id = np.array(new_id(sediment_id_old, id_change))
topsoil_ob = np.log(topsoil[:,0])
sediment_ob = np.log(sediment[:,0])
disM = distanceM(topsoil[:,1], topsoil[:,2])
disM = np.array(disM)

basic_model = pm.Model()

with basic_model:
    alpha = pm.Normal('alpha', mu=0, sd=1000000, testval=np.random.randn())
    beta  = pm.Normal('beta', mu=0, sd=1000000, testval=np.random.randn())
    sigma = pm.InverseGamma('sigma', alpha=0.001, beta=0.001,testval=np.random.randn()**2)
    phi = pm.InverseGamma('phi', alpha=0.001, beta=0.001, testval=np.random.randn()**2)
    cov =  -sigma*np.exp(-disM)
    mu = np.zeros(len(topsoil_id))
    beta_s = pm.MvNormal('beta_s', mu, cov, shape=len(topsoil_id),testval=np.random.randn(len(topsoil_id)))
    w_T = pm.InverseGamma('omegaT', alpha=0.001, beta=0.001, testval=np.random.randn())
    w_s = pm.InverseGamma('omegaS', alpha=0.001, beta=0.001, testval=np.random.randn())
    gam = pm.Uniform('gamma', lower=0, upper=0.1633, testval=np.random.randn()**2)
    I = np.eye(len(adj) )
    tau_s = pm.InverseGamma('tau_s', alpha=0.001, beta=0.001, testval=np.random.randn()**2)
    mu_s = pm.MvNormal('mu_s', np.zeros(len(adj)), 1000000*np.ones((len(adj), len(adj))), shape=len(adj))
    cov_temp = I - gam*adj
    invA = pm.math.matrix_inverse(cov_temp)
    cov_s =  tau_s*invA
    y_s = pm.MvNormal('y_s', mu_s, cov_s, shape=len(adj), testval=np.random.randn(len(adj)))
    id_temp = np.add(topsoil_id, -1)
    mu_t = alpha + np.dot((beta+ beta_s), y_s[id_temp])
    tau_t = pm.InverseGamma('tau_t', alpha=0.001, beta=0.001, testval=np.random.randn())
    cov_t = tau_t*np.eye(len(topsoil_id))
    y_t = pm.MvNormal('y_t', mu_t, cov_t, shape=len(topsoil_id), testval=np.random.randn(len(topsoil_id)))
    z_t = pm.MvNormal('z_t', y_t, w_T*np.eye(len(topsoil_id)),  observed=topsoil_ob,total_size=topsoil_ob.shape[0])
    id_temp = np.add(sediment_id, -1)
    mu_ob_s = y_s[id_temp]
    z_s = pm.MvNormal('z_s', mu_ob_s, w_s * np.eye(len(sediment_id)),  observed=sediment_ob,total_size=sediment_ob.shape[0])
    step = pm.Slice()
    trace = pm.sample(draws=5000,step=step)
    pm.summary(trace)