import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import random
from scipy import stats

class ModelOne:
    def __init__(self,N,tau,q,g):
	self.N = N
	self.tau = tau
	self.q = q
	self.g = g
	self.M = np.zeros(N)
	self.pos = 0
	self.counts = np.zeros(N)
	self.impurity = int(N/2)
	self.M[self.impurity] = 1.
	self.counts[self.impurity] = 1.
	self.cstm = stats.rv_discrete(name='cstm',values=(np.arange(self.N),self.M))

    def set_q(self,q):
	self.q = q

    def set_g(self,g):
	self.g = g

    def set_tau(self,tau):
	self.tau = tau

    def reset(self):
	self.M = np.zeros(self.N)
	self.pos = 0
	self.counts = np.zeros(self.N)
	self.M[self.impurity] = 1.
	self.counts[self.impurity] = 1
	self.cstm = stats.rv_discrete(name='cstm',values=(np.arange(self.N),self.M))

    def update(self):
	r = random()
	if self.pos != self.impurity:
	    if r < (1.-self.q)/2.:
		self.pos = pbcIndex(self.pos+1,self.N)
	    elif r < (1.-self.q):
		self.pos = pbcIndex(self.pos-1,self.N)
	    else:
		self.pos = self.cstm.rvs()
	else:
	    if r < (1.-self.g-self.q*(1.-self.g))/2.:
		self.pos = pbcIndex(self.pos+1,self.N)
	    elif r < (1.-self.g - self.q*(1-self.g)):
		self.pos = pbcIndex(self.pos-1,self.N)
	    elif r < (1.-self.g):
		self.pos = self.cstm.rvs()
	    else:
		self.pos = self.pos
	self.M /= np.exp(-1./self.tau)
	self.M[self.pos] += 1.
	self.M /= np.sum(self.M)
	self.counts[self.pos] += 1
	self.cstm = stats.rv_discrete(name='cstm',values=(np.arange(self.N),self.M))

    def simulate(self, T):
	for i in range(T):
	    self.update()
	return (self.M,self.counts)


def pbcIndex(n,size=40):
    if n >= size:
        return n % size
    if n == -1:
        return size - 1
    return n

def plotCounts(c,saveStr='M1_distrib.png'):
    n = len(c)
    c /= np.sum(c)
    plt.plot(np.arange(n)-int(n/2),c,marker='.')
    plt.xlabel('Position (x)')
    plt.ylabel('P(x)')
    plt.savefig(saveStr,bbox_inches='tight')
    plt.close()

def mapP0q(N,T,tau,g,numPoints=20,saveStr="M1_P0_sample.png"):
    m = ModelOne(N,tau,0.,g)
    qArr = np.linspace(0.,1.,numPoints)
    P0 = np.zeros(numPoints)
    for i,q in enumerate(qArr):
	m.reset()
	m.set_q(q)
	(mDat,cDat) = m.simulate(T)
	cDat /= np.sum(cDat)
	P0[i] = cDat[m.impurity]
    plt.plot(qArr,P0,marker='.')
    plt.xlabel('q')
    plt.ylabel(r'$P_0$')
    plt.title("g="+str(g)+", tau=" + str(tau))
    plt.savefig(saveStr,bbox_inches='tight')

if __name__== "__main__":
    m = ModelOne(40,10,.3,.5)
    # (memDat,countDat) = m.simulate(10000)
    # plotCounts(countDat,saveStr="M1_sample.png")
    mapP0q(40,10000,10,.5,20)
