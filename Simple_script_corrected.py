import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.mlab as mlab
import math

## Define the parameters

#Number of neurons
N=360

# Strength of excitation
GEE=6	 	# strength of excitation to excitatory neurons
GEI=4		# strength of excitation to inhibitory neurons
GIE=3.4
GII=0.85

#Initial currents
Ie0=0.2
Ii0=0.5


#Firing-rate             
sigmae=1
sigmai=3
taue=20
taui=10


#Excitatory matrix
WEE=np.zeros((N,N))
window = signal.gaussian(N, std=100)
window=window*(-1)+1  #reversed gaussian
for i in range(len(WEE)):
    for j in range(len(WEE)):
        WEE[i,j]=window[abs(i-j)] 
WEE=WEE/100



#Stimulus
stimul=np.zeros((N,N))
window = signal.gaussian(N, std=30)
for i in range(len(stimul)):
    for j in range(len(stimul)):
        stimul[i,j]=window[abs(i-j)] 
stimul=stimul*1.5

#Initialization of variables
stock_ex=np.zeros(N)
stock_in=np.zeros(N)
re=np.zeros(N)               
ri=np.zeros(N) 
Ie=0
Ii=0

#Time
deltat = 0.02 # 1/100 of the main time scale of the system
T = 42. 
t = np.linspace (0, T, int(T/deltat)) 
print(len(t)-1100)

## Bump_attractor

# Input-output function
def transfer(x):
    if x<0:
        x=0
    elif x>0 and x<1:
        x=x*x
    elif x>=1:
        x=np.sqrt(4*x-3)   
    return(x)
    
x_range = np.linspace(-15,100,1e5)
f_range = [transfer(i) for i in x_range]


#Simulate
def model(id):
    global stock_ex
    global stock_in
    global re
    global ri
    global Ie
    global Ii
    
    for i in range (len(t)-1100):
        
        #the four steps
        if i>=0 and i<100:
            stimulus=np.zeros(N)
        if i>=200 and i<300:
            stimulus=stimul[id]     #here to change the stimulus !!
        if i>=300 and i<900:
            stimulus=np.zeros(N)
        if i>=900 and i<1000:
            stimulus=-stimul[id]   #here to change the stimulus !!
    
        
        #store the firing rate
        stock_ex=np.vstack([stock_ex,re])
        stock_in=np.vstack([stock_in,ri])
        
        #update values of excitatory neurons
        Ie=GEE*np.matmul(WEE,re)+(Ie0-GIE*np.mean(ri))*np.ones(N)+stimulus
        #gaussian_noise_ex=sigmae*np.random.randn(N)
        gaussian_noise_ex=np.sqrt(deltat/taue)*sigmae*np.random.randn(N)
        re=re+deltat/taue*(-re+np.interp(Ie,x_range,f_range)+gaussian_noise_ex)
        
        #update values of inhibitory neurons
        Ii=(GEI*np.mean(re)-GII*np.mean(ri)+Ii0)*np.ones(N)
        #gaussian_noise_inh=sigmai*np.random.randn(N)
        gaussian_noise_inh=np.sqrt(deltat/taui)*sigmai*np.random.randn(N)
        ri=ri+deltat/taui*(-ri+np.interp(Ii,x_range,f_range)+gaussian_noise_inh) 
        
        #print(len(stock_ex))
    
model(320)
#shift of 180=> stimul(180) match to a max at 0 and stimul(320) match to a max of 140


## Plot all excitatory neurons firing rate
plt.plot(stock_ex) 
plt.show()
#plt.close()

#Save data (stock_ex)
# address_csv='/home/lucdufour/Documents/Cogmaster/Cours/S3/CA6/Project/Bump_attractor_model/Plot/essai.csv'
# print(address_csv)
# np.savetxt(address_csv, stock_ex, delimiter=",")

## Plot the firing-rate for a range of time for all neurons (=bump for a range of time)
for i in range(900,1000):  #range of time steps
    plt.plot(stock_ex[i])
plt.show()
#plt.close()



##Plot the evolution of the bump
bump=[]
for i in range(0,1000):
    bump.append(np.argmax(stock_ex[i]))
plt.plot(bump)
plt.show()


## Plot the tuning curve of a neuron
#Principle: compute the firing-rate at a constant time (after the stimulus) for a given neuron for all stimulus (ie all possible angles for the maximum value of the stimulus)

#Compute
tuning=[]
for i in range(len(stimul)):
    
    #Initialization of variables
    stock_ex=np.zeros(N)
    stock_in=np.zeros(N)
    re=np.zeros(N)               
    ri=np.zeros(N) 
    Ie=0
    Ii=0
    
    #compute firing-rate
    print(i)  #follow evolution of the algorithm
    model(i)
    tuning.append(stock_ex[500][180])  #500 = constant time where we look at the firing rate of the neuron, 50 = number/angle of the neuron studied
    

#Plot
plt.plot(tuning)
plt.show()

#Save plot
# address_png='/home/lucdufour/Documents/Cogmaster/Cours/S3/CA6/Project/Bump_attractor_model/Plot/tuningcurve'+'_500'+'_180'+'.png'
# print(address_png)
# plt.savefig(address)

#Save data (tuning)
# address_csv='/home/lucdufour/Documents/Cogmaster/Cours/S3/CA6/Project/Bump_attractor_model/Plot/tuningcurve'+'_500'+'_180'+'.csv'
# print(address_csv)
# np.savetxt(address_csv, tuning, delimiter=",")


