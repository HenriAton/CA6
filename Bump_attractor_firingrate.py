import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

## Define the parameters

# Firing-rates
rmin=np.random.normal(7, 1)         #gaussian distribution for rmin and rmax
rmax=np.random.normal(14, 3.4)

while rmax<rmin:                          #rmin has to be < rmax
    rmin=np.random.normal(7, 1)
    rmax=np.random.normal(14, 3.4)


# Width of the bump
w=35
w0=w

# Angles
tetas=[0,45,90,135,180,225,270,315]    #'free' parameter
tetai=250    #'free' parameter

# Gaussian white noise
sigma=4.04
mean = 0
std = 1

# Time-evolution
tau = 10. # ms


## Compute the time-varying firing rate of a neuron

# Compute the time evolution
deltat = 0.001   # very small, otherwise t very big and n=0
T = 3. 
t = np.linspace (0, T, int(T/deltat))

# Width of the bump
k=180**2/(np.pi**2*w**2)  #steady for the bump attractor model
k0=k


# Computation of angle
#n=(1/(std*np.sqrt(2*np.pi)))*np.exp(-(t**2/(2*(std**2))))
#teta= tetas + sigma*integrate((1/(std*np.sqrt(2*np.pi)))*np.exp(-(t**2/(2*(std**2)))), n)
#n=numpy.random.normal(0, 1)*t #t can't go in the brackets, because n input has to be < 32

teta= tetas[1] + sigma*np.sqrt(np.pi)*t   #not a white gaussian noise => raise linearly with time  !!!
#teta= tetas + sigma*n

# Firing-rate

r=rmin +(rmax-rmin)*((k*np.exp(np.cos(tetai-teta))-np.exp(-k0))/(np.exp(k0)-np.exp(-k0)))


# Plot the result 

plt.plot(t, r, label='Try')

plt.xlabel('$t$ (ms)')
plt.ylabel('$r$ (Hz)')

plt.show()


## Mimick the experimental data

# Fixation period of 1 s: Homogeneous Poisson at rate rmin
tfixation = np.linspace (0, 1, int(1/0.001))

rfixation = np.zeros( len(tfixation) )     #create a list of fixed value of rfixation, of size tfixation
rfixation[0]=rmin                           # replace by a poisson spike generator !!!
for i in range ( len(tfixation) -  1 ):
    rfixation[i+1] = rfixation[i]

# Cue period of 0.5 s : #θ(t) = θs
tcue = np.linspace (0, 0.5, int(0.5/0.001)) 

rcue = np.zeros( len(tcue) )     #create a list of fixed value of rcue, of size tcue
rcue[0]=rmin +(rmax-rmin)*((k*np.exp(np.cos(tetai-tetas[1]))-np.exp(-k0))/(np.exp(k0)-np.exp(-k0)))
for i in range ( len(tcue) -  1 ):
    rcue[i+1] = rcue[i]           # not continous, add a transition from rfixation to rcue?  !!!


# Delay period of 3 s : time-varying firing rate
tdelay = np.linspace (0, 3, int(3/0.001))  
tetadelay= tetas[1] + sigma*np.sqrt(np.pi)*tdelay    #not a white gaussian noise !!!

rdelay=rmin +(rmax-rmin)*((k*np.exp(np.cos(tetai-tetadelay))-np.exp(-k0))/(np.exp(k0)-np.exp(-k0)))
#not continous, it has to vary around rcue: replace rmin or rmax by rcue[len(tcue)-1]  doesn't work. Only change between rdelay and rcue: tetadelay is replaced by tetas. So, in order to have rdelay=rcue at t=O, tdelay[0] has to be =0 and not =1.5



# Plot the result 

plt.plot(tfixation, rfixation, label='Fixation')
plt.plot(tcue+1, rcue, label='Cue')
plt.plot(tdelay+1.5, rdelay, label='Delay')

plt.xlabel('$t$ (ms)')
plt.ylabel('$r$ (Hz)')

plt.legend(loc = 4)

plt.show()


## Draft

# Calcul symbolique
from sympy import *
x = symbols('x')
F = integrate(sqrt(x+2),x)
print(F)

# Calcul intégral
from scipy.integrate import quad
def integrand(x, a, b):
     return a*x**2 + b
a = 2
b = 1
I = quad(integrand, 0, 1, args=(a,b))

# Gaussian noise
n=numpy.random.normal(mean, std)
n = np.zeros( len(t) )
n=numpy.random.normal(mean, std,t)


# Poisson spike generator
#https://stackoverflow.com/questions/36050119/simulating-a-neuron-spike-train-in-python
import scipy as sp
import numpy as np
import pylab as plt

def poisson_spikes(t, N=100, rate=1.0 ):
    spks = []
    dt = t[1] - t[0]
    for n in range(N):
        spkt = t[np.random.rand(len(t)) < rate*dt/1000.] #Determine list of times of spikes
    return spkt
        idx = [n]*len(spkt) #Create vector for neuron ID number the same length as time
        spkn = np.concatenate([[idx], [spkt]], axis=0).T #Combine tw lists
        if len(spkn)>0:        
            spks.append(spkn)
    spks = np.concatenate(spks, axis=0)
    return spks

dt = 0.001
t = sp.arange(0.0, 100.0, dt) #The time to integrate over
#ic = [-65, 0.05, 0.6, 0.32]

N = 1
spks =  poisson_spikes(t, N, rate=rmin)
print(spks)


