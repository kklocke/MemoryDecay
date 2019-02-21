import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import random
# from random import choices, random
from scipy import stats

T = 100000
impuritySite = 20
q = .3
g = .5
tau = .1
N = 40
# for now we'll implement with PBC

def pbcIndex(n,size=40):
    if n >= size:
        return n % size
    if n == -1:
        return size - 1
    return n

M = np.zeros(N)
M[0] = 1.;
counts = np.zeros(N)
counts[0] = 1;
pos = 0

for t in range(T):
    r = random()
    cstm = stats.rv_discrete(name='cstm',values=(np.arange(N),M/np.sum(M)))
    if pos != impuritySite:
        if r < (1.-q)/2.:
            pos = pbcIndex(pos+1)
        elif r < (1.-q):
            pos = pbcIndex(pos-1)
        else:
            pos = cstm.rvs() # choices([i for i in range(20)], weights=M)
    else:
        if r < (1.-g-q+g*q)/2.:
            pos = pbcIndex(pos+1)
        elif r < (1.-q-g+q*g):
            pos = pbcIndex(pos-1)
        elif r < (1.-g):
            pos = cstm.rvs() # choices([i for i in range(20)],weights=M)
        else:
            pos = pos
    M *= np.exp(-1/tau)
    counts[pos] += 1
    M[pos] += 1

plt.plot([i for i in range(N)], counts)
plt.savefig('distrib.png')
plt.close()
