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
D = np.diag(adj.sum(axis=1))
basic_model = pm.Model()
print (sediment_id)
with basic_model:
    tau = pm.Gamma('tau', alpha =2.0 ,beta=2.0)
    alpha = pm.Uniform('alpha', lower=0, upper=1)
    phi = pm.MvNormal('phi', mu = 0, tau = tau*(D - alpha*adj), shape=(1, len(adj)))
    mu = pm.Deterministic('mu', phi.T)
    tau_s = pm.InverseGamma('tau_s', alpha=0.001, beta=0.001, testval=np.random.randn()**2)
    z_s = pm.Normal('z_s', mu = mu[sediment_id-1], sd= tau_s ,  observed=sediment_ob)
    trace = pm.sample(3e3, njobs=1)
    pm.summary(trace)
    pm.traceplot(trace)
