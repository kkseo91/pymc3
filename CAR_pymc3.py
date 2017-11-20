import pickle # python3
import theano
floatX = theano.config.floatX
import pymc3 as pm
from scipy import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import theano.tensor as tt
from warnings import filterwarnings
filterwarnings('ignore')


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
D = np.diag(adj.sum(axis=1))
basic_model = pm.Model()
print (D[0])
print (len(adj))
print (np.where( sediment_id == 5)[0])
sd_ob = np.random.randn(274)
d={}
with basic_model:
    tau_s = pm.Gamma('tau_s', alpha =2,beta=2, testval = 3)
    gamma = pm.Uniform('gamma', lower=0, upper=0.15)
    mu_s = pm.Normal('mu_s', mu=0, sd =1000, testval =1.0)
    y_s = pm.MvNormal('y_s', mu = mu_s*np.ones(274), tau = tau_s*(D - gamma*adj), shape=len(adj))
    w_s = pm.InverseGamma('omegaS', alpha=0.001, beta=0.001, testval=np.random.randn()**2)
    for i in range(274):
        index = np.where( sediment_id == i)[0]
        d["string{0}".format(i)] = pm.Normal('Yi'+str(i), mu=y_s[i],sd = w_s**0.5, observed=sediment_ob[index], total_size=len(sediment_ob[index]))
    alpha = pm.Normal('alpha', mu=0, sd=1000, testval=np.random.randn())
    beta  = pm.Normal('beta', mu=0, sd=1000, testval=np.random.randn())
    sigma = pm.Gamma('sigma', alpha=1, beta=1,testval=np.random.randn()**2)
    phi = pm.Gamma('phi', alpha=1, beta=1, testval=np.random.randn()**2)
    cov =  np.exp(-disM/phi)*sigma
    mu = np.zeros(len(topsoil_id))
    beta_s = pm.MvNormal('beta_s', mu, cov, shape=len(topsoil_id),testval=np.random.randn(len(topsoil_id)))
    w_T = pm.InverseGamma('omegaT', alpha=0.001, beta=0.001, testval=np.random.randn()**2)
    id_temp = np.add(topsoil_id, -1)
    mu_t = alpha + np.dot((beta+ beta_s), y_s[id_temp])
    tau_t = pm.InverseGamma('tau_t', alpha=0.001, beta=0.01, testval=1)
    cov_t = tau_t*np.eye(len(topsoil_id))
    y_t = pm.MvNormal('y_t', mu_t, cov_t, shape=len(topsoil_id), testval=np.random.randn(len(topsoil_id)))
    z_t = pm.MvNormal('z_t', y_t, w_T*np.eye(len(topsoil_id)),  observed=topsoil_ob,total_size=topsoil_ob.shape[0])
    step = pm.Metropolis()
    trace = pm.sample(60000,step = step)
    pm.summary(trace)
    dataframe = pm.trace_to_dataframe(trace)
    dataframe.to_csv('result.csv', sep='\t')
with open('sampling_result.pkl', 'wb') as buff:
    pickle.dump({'model': basic_model, 'trace': trace}, buff)

'''
    tau = pm.Gamma('tau', alpha =2.0 ,beta=2.0)
    alpha = pm.Uniform('alpha', lower=0, upper=0.5)
    phi = pm.MvNormal('phi', mu = 0, tau = tau*(D - alpha*adj), shape=(1, len(adj)))
    mu = pm.Deterministic('mu', phi.T)
    y_s = pm.MvNormal('y_s', mu=mu.ravel(), cov=np.eye(274))
    mu_temp = pm.Deterministic('mu_tem', y_s.ravel())
    z_s = pm.MvNormal('z_s', mu = mu_temp[sediment_id-1], cov=np.ones(6025),  observed=sediment_ob)
    trace = pm.sample(3e3, njobs=1)
    pm.summary(trace)
    pm.traceplot(trace)
'''
 
