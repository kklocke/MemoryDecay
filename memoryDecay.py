import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import random
from scipy import stats
import sys

class ModelOne:
    def __init__(self,N,tau,q,g):
	self.N = N
	self.tau = tau
	self.q = q
	self.g = g
	self.M = np.zeros(N)
	self.counts = np.zeros(N)
	self.impurity = int(N/2)
	self.pos = self.impurity
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
	self.pos = self.impurity
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


class ModelOne_2D:
    def __init__(self):
	self.pos = (0,0)
	self.impur = (0,0)
	self.M = {(0,0):1}
	self.counts = {(0,0):1}
	self.tau = 1000
	self.cstm = stats.rv_discrete(name="cstm",values=(np.arange(len(self.M.keys())),self.M.values()))
	self.g = .9
	self.q = .3

    def __init__(self,tau,g,q):
	self.pos = (0,0)
	self.impur = (0,0)
	self.M = {(0,0):1}
	self.counts = {(0,0):1}
	self.tau = tau
	self.g = g
	self.q = q
	self.cstm = stats.rv_discrete(name="cstm",values=(np.arange(len(self.M.keys())),self.M.values()))

    def set_tau(self,tau):
	self.tau = tau

    def set_q(self,q):
	self.q = q

    def set_g(self,g):
	self.g = g

    def reset(self):
	self.pos = (0,0)
	self.M = {(0,0):1}
	self.counts = {(0,0):1}
	self.cstm = stats.rv_discrete(name="cstm",values=(np.arange(len(self.M.keys())),self.M.values()))

    def update(self):
	r = random()
	# print(self.pos)
	# print(self.M.keys())
	if (self.pos[0] != self.impur[0]) or (self.pos[1] != self.impur[1]):
	    if r < (1.-self.q)/4.:
		self.pos = (self.pos[0]+1,self.pos[1])
	    elif r < (1.-self.q)/2.:
		self.pos = (self.pos[0]-1,self.pos[1])
	    elif r < 3.*(1.-self.q)/4.:
		self.pos = (self.pos[0],self.pos[1]+1)
	    elif r < (1.-self.q):
		self.pos = (self.pos[0],self.pos[1]-1)
	    else:
		self.pos = self.M.keys()[self.cstm.rvs()]
	else:
	    if r < (1.-self.g -self.q*(1.-self.g))/4.:
		self.pos = (self.pos[0]+1,self.pos[1])
	    elif r < (1.-self.g-self.q*(1.-self.g))/2.:
		self.pos = (self.pos[0]-1,self.pos[1])
	    elif r < 3.*(1.-self.g-self.q*(1.-self.g))/4.:
		self.pos = (self.pos[0],self.pos[1]+1)
	    elif r < (1.-self.g-self.q*(1.-self.g)):
		self.pos = (self.pos[0],self.pos[1]-1)
	    elif r < (1.-self.g):
		self.pos = self.M.keys()[self.cstm.rvs()]
	    else:
		self.pos = self.pos
	for k in self.M.keys():
	    self.M[k] *= np.exp(-1./self.tau)/(1. + np.exp(-1./self.tau))
	if self.pos in self.M:
	    self.M[self.pos] += 1./(1. + np.exp(-1./self.tau))
	    self.counts[self.pos] += 1.
	else:
	    self.M[self.pos] = 1./(1. + np.exp(-1./self.tau))
	    self.counts[self.pos] = 1.
	self.cstm = stats.rv_discrete(name="cstm",values=(np.arange(len(self.M.keys())),self.M.values()))

    def simulate(self,T):
	for i in range(T):
	    self.update()
	return float(self.counts[self.impur])/float(T)


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
    fig,ax = plt.subplots()
    for i,q in enumerate(qArr):
	m.reset()
	m.set_q(q)
	(mDat,cDat) = m.simulate(T)
	cDat /= np.sum(cDat)
	P0[i] = cDat[m.impurity]
	ax.plot([j for j in range(N)], cDat, label=str(q))
    ax.set_xlabel("position")
    ax.set_ylabel("probability")
    plt.savefig("histo_" + saveStr, bbox_inches='tight')
    plt.close('all')
    plt.plot(qArr,P0,marker='.')
    plt.xlabel('q')
    plt.ylabel(r'$P_0$')
    plt.title("g="+str(g)+", tau=" + str(tau))
    plt.savefig(saveStr,bbox_inches='tight')
    plt.close('all')

if __name__== "__main__":
    # m = ModelOne(40,10,.3,.5)
    # (memDat,countDat) = m.simulate(10000)
    # plotCounts(countDat,saveStr="M1_sample.png")
    # mapP0q(40,30000,1000,.9,20)
    vals = sys.argv[1:]
    if len(vals) == 0:
	mapP0q(40,1000,10,.9,20)
    else:
	assert(len(vals) == 6)
	N = int(vals[0])
	T = int(vals[1])
	tau = float(vals[2])
	g = float(vals[3])
	grid = int(vals[4])
	fname = vals[5] + ".png"
	assert(N > 0)
	assert(T > 0)
	assert(tau > 0)
	assert((g >= 0) and (g <= 1))
	assert(grid > 0)
	assert(fname != ".png")
	mapP0q(N,T,tau,g,grid,saveStr=fname)
