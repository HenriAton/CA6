import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal

## Define the parameters

#Parameters model
#Number of neurons
N=360

# All-to-all connectivity matrices
W0=1  #not given in the paper

# Strength of excitation
GEE=6	 	# strength of excitation to excitatory neurons
GEI=4		# strength of excitation to inhibitory neurons
GIE=3.4
GII=0.85


#Initial currents
Ie0=0.2
Ii0=0.5

# Input-output function
Im=0

#Initial firing-rates
re=np.zeros(N)               
ri=np.zeros(N) 

#Firing-rate             
sigmae=1
sigmai=3
taue=20
taui=10

#Stimulus
stimulus=np.ones(N)*100






#Parameters simulation
#Time
deltat = 2
T = 4200. # ms
t = np.linspace (0, T, int(T/deltat))

#Realtime(): Initialization of store variables
stock_ex=np.zeros(N)
stock_in=np.zeros(N)

#Sim(): Steps
tstep=500
nbstep=3
nbneuron=1

#variator()
stepsim=1
lim=5

#plot_var_param(): given-time
given_time=10



## Build the model

# Connectivity matrices
WII=WIE=np.ones((N,N))
WEI=-WIE #inhibition  =>useless


def ex_matrix(x):
    WEE=np.zeros((N,N))
    window = signal.gaussian(N, std=100)
    window=window*(-1)+1  #reversed gaussian
    for i in range(len(WEE)):
        for j in range(len(WEE)):
            WEE[i,j]=window[abs(i-j)] 
    WEE=WEE/x
    return(WEE)

WEE=ex_matrix(100)


# Input-output function (Phi)
def transfer(x):
    if x<0:
        x=x*x
    elif x>0 and x<1:
        x=x
    elif x>=1:
        x=np.sqrt(4*x-3)   
    return(x)


#Phi output    
def phi(y):
    for i in range(len(y)):
        y[i]=transfer(y[i])
    return(y)


# Simulate

#One step
def realtime(t, stim):
    for i in range (t): #for i in range ( len(t) -  1 ):
    
        #take input
        global stock_ex
        global stock_in
        global Ie
        global Ii
        global re
        global ri
        
        #build the list
        stock_ex=np.vstack([stock_ex,re])
        stock_in=np.vstack([stock_in,ri])
        
        #update values of excitatory neurons
        Ie=GEE*np.matmul(WEE,re)+(Ie0-GIE*np.mean(ri))*np.ones(N)+stim
        gaussian_noise_ex=sigmae*np.random.randn(N) #ok
        re=re+deltat/taue*(-re+phi(Ie)+gaussian_noise_ex) #addition of phi(Ie) and gaussian_noise-_ex so the 
        
        #update values of inhibitory neurons
        Ii=(GEI*np.mean(re)-GII*np.mean(ri)+Ii0)*np.ones(N)
        gaussian_noise_inh=sigmai*np.random.randn(N)
        ri=ri+deltat/taui*(-ri+phi(Ii)+gaussian_noise_inh) 
    return
  
#The four steps
def sim():
    #take input
    global tstep
    global nbstep
    
    #execute steps
    if nbstep>0:
        realtime(tstep, 0)
    if nbstep>1:
        realtime(tstep, stimulus)
    if nbstep>2:
        realtime(tstep,0)
    if nbstep>3:
        realtime(tstep, -stimulus)
    return
    
#Plot xth excitatory neuron and xth inhibitory neuron
def plot_neuron(x):
    ne=np.zeros(tstep*nbstep)   #not efficient
    ni=np.zeros(tstep*nbstep)
    for i in range(tstep*nbstep):
        ne[i]=stock_ex[i][x]
        ni[i]=stock_in[i][x]
    plt.plot(ne)
    plt.plot(ni)
    return


#Simulate and plot
def total():
    sim()
    plt.figure(1)
    plt.title(str(N)+' excitatory neurons')
    plt.plot(stock_ex)
    plt.figure(2)
    plt.plot(stock_in)
    plt.title(str(N)+' inhibitory neurons')
    plt.figure(3)
    plot_neuron(nbneuron)
    plt.title(' Excitatory and inhibitory neurons n°'+str(nbneuron))
    plt.show()
    return


#Plot the firing rate according to time according to different strenghts of the parameter
def fire_stim(keep):
    count=0
    for i in range(int(lim/stepsim)):
        count+=stepsim
        plt.plot(keep[i], label = str(count))
    plt.legend(loc = 4)
    plt.show()
    return

#Plot at a given time the firing rate according to the strength of the parameter
def plot_var_param(given_time):
    ne=[]
    ni=[]
    absciss=np.arange(stepsim, lim+stepsim, stepsim)
    for i in range(int(lim/stepsim)):
        ne.append(keep_ex[i][given_time])
        ni.append(keep_in[i][given_time])
    plt.plot(absciss, ne)
    plt.plot(absciss, ni)
    return

##Simulate and plot
#Simulate
sim()

#Plot all excitatory neurons firing rate
plt.plot(stock_ex) #works => very nice figure +++
plt.show()
   
#Plot all inhibitory neurons firing rate
plt.plot(stock_in)
plt.show()

#Compare the two
plt.figure(1) #to let the index start at 1
plt.plot(stock_ex)
plt.figure(2)
plt.plot(stock_in)
plt.show()

#Plot xth excitatory neuron and xth inhibitory neuron
plot_neuron(nbneuron)
plt.show()
    

#Simulate and plot
total()



#Stimulus
theta = [0:N-1]/N*2*pi;
theta=theta-pi;
v = exp(kappa*cos(theta));
v = v/sum(v);
stimulus = stim*v'


## Analysis: change the strength of a parameter

# Change the strength of the parameter
def variator(varia, zero): #zero=0 or zero=np.zeros(N)
    
    #take input
    global keep_ex
    global keep_in
    global stock_ex  #even if I don't modify the variable, I have to call it with global +++
    global stock_in
    
    global WEE  # vary with the parameter
    
    #variation
    for e in range(int(lim/stepsim)):
        
        #reinitialize essential parameters
        re=np.zeros(N)               
        ri=np.zeros(N)
        stock_ex=np.zeros(N)
        stock_in=np.zeros(N)
        
        
        #reinitialize optionnal parameters (to play with)
        GEE=6	 	
        GEI=4	
        GIE=3.4
        GII=0.85
        Ie0=0.2
        Ii0=0.5
        stimulus=np.ones(N)*100
        WEE=ex_matrix(100)
        
        
        #parameters of variator
        row_ex=[]
        row_in=[]
        
        #vary
        varia=zero
        varia=stepsim+e*stepsim
        WEE=varia              # vary with the parameter
        
        #execution
        sim()
        for j in range(tstep*nbstep):
            row_ex.append(np.mean(stock_ex[j]))
            row_in.append(np.mean(stock_in[j]))
        print(row_ex)      #speaker here
        keep_ex=np.vstack([keep_ex,row_ex])
        keep_in=np.vstack([keep_in,row_in])
    return
    
#quick variator
def quick_variator(stepsi, li):
    #take input
    global stepsim
    global lim
    global keep_ex
    global keep_in
    
    #modify stepsim and lim for the plot
    stepsim=stepsi
    lim=li
    
    #reinitialize keep_ex and keep_in
    keep_ex=np.zeros(tstep*nbstep)
    keep_in=np.zeros(tstep*nbstep)
    
    #variator
    variator(WEE, ex_matrix(100)  #to vary
    return
    

quick_variator(1,5)
quick_variator(50, 250)
    


#Plot the firing rate according to time according to different strenghts of the stimulus
fire_stim(keep_ex)    #bug: doesn't plot all the curbs or overlapped curbs ?
fire_stim(keep_in)

#Plot at a given time the firing rate according to the strength of the stimulus
plot_var_param(given_time)
plt.show()



## Draft
#2 possibles errors : variable is not global, so it stays at its inital value. Or sim doesn't change according to the variable because ot doesn't take it in input => I have to chek that.

#Call a variable with global: if I modify after calling this variable, this modification is avaible everywhere,included recursively in the function
a=34
def check():
    global a
    print(a)
    a=2
    print(a)
    rer()
    return
    #a=rer()
    #return(a)
check()
    
def rer():
    global a
    print(a)
    return(a)

rer()

#Bug
def var_stim(stepsim,lim):
    global keep
    for i in range(int(lim/stepsim)):
        re=np.zeros(N)               
        ri=np.zeros(N)
        Ie0=0.2
        Ii0=0.5
        stock_ex=np.zeros(N)
        stock_in=np.zeros(N)
        
        row=[]
        stimulus=np.zeros(N)+stepsim
        sim(tstep, nbstep)
        for i in range(len(stock_ex)-1):
            row.append(np.mean(stock_ex[i]))
        keep=np.vstack([keep,row])
    return

lim=150
stepsim=50
keep=np.zeros(tstep*nbstep)
for i in range(int(lim/stepsim)):
    re=np.zeros(N)               
    ri=np.zeros(N)
    Ie0=0.2
    Ii0=0.5
    stock_ex=np.zeros(N)
    stock_in=np.zeros(N)
    
    row=[]
    stimulus=np.zeros(N)+stepsim
    sim(tstep, nbstep)
    for i in range(len(stock_ex)-1):
        row.append(np.mean(stock_ex[i]))
    keep=np.vstack([keep,row])
     
print(len(row))
print(len(keep))
        
row=[]
for i in range(len(stock_ex)):
    row.append(np.mean(stock_ex[i]))
print(row)
print(stock_ex)
print(len(row))

#Phi input
#Ie=GEE*np.matmul(WEE,re)+(Ie0-GIE*np.mean(ri))*np.ones(N)   #be careful with matrices calculus => if I don't use np.matmul, can do errors without signaling it => for example a matrix (WEE) multiplied by an array (re) gives a matrix instead of an array
#Ii=(GEI*np.mean(re)-GII*np.mean(ri)+Ii0)*np.ones(N)


#Can't modify global variables in the script
stimulus=np.ones(N)*200
def try():
    stimulus=np.ones(N)*200
    for in range(5):
        essai()
    return
    
def essai():
    stimulus=stimulus+1
    return
essai()
#Useless packages
import networkx as nx              # module useful to manipulate graphs such as connectivity matrices
from scipy.fftpack import fft, fftshift

#Print first neuron firing rate
r1=np.zeros(3)
re=np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
for i in range(len(re-1)):
    r1[i]=re[i][0]
print(r1)

#Append list vertically
stock=np.zeros(5)
for i in range(4):
    stock=np.vstack([stock,np.ones(5)])
print(stock)

#my way

Ie=WEE[1]*re0*GEE-WEI[1]*ri0*GIE+Ie0
Ii=WIE[1]*re0*GEI-WII[1]*ri0*GII+Ii0


#sum
def output_neurons_in(u):
    output_inh=0
    for v in range(len(WIE)):
        output_inh+=WIE[u,v]*1*GEI-WII[u,v]*1*GII
    return(int(output_inh))
ess=output_neurons_in(2)
print(ess)
print((4-0.85)*360)

#transfer
a=transfer(output_neurons_in(1))
print(a)

#firing-rate
#re[0]=0?
print(ri)
for i in range ( len(t) -  1 ):
    re[i+1]=re[i]+deltat/T*(re[i]+transfer(output_neurons_in(i)+Ie0)+gaussian_noise_ex)

for i in range ( len(t) -  1 ):
    print(transfer(output_neurons_in(i)))

#Initial currents
Ie0par=0.5
Ii0par=0.2

# Initial currents
Ie0=np.ones(N)*Ie0par   #initial current for excitatory neurons
Ii0=np.ones(N)*Ii0par   #initial current for inhibitory neurons

# Firing-rate
re=(re0-phie-sigmae*epsilon0)*np.exp(-t/taue)+phie+sigmae*epsilon
ri=(ri0-phii-sigmai*epsilon0)*np.exp(-t/taui)+phii+sigmai*epsilon

#Calculus with matrices
print(np.ones((5,5))*np.ones(5))
print(np.matmul(np.ones((5,5)),np.ones(5)))
print(np.ones(5))
print(np.ones((1,5))*np.ones((5,1)))

#Phi input
Ie=GEE*WEE-GEI*WEI+Ie0+Im
Ii=GIE*WIE-GII*WII+Ii0

#Connectivity matrices using module network

# All-to-all connectivity matrices

G=nx.complete_graph(10)  # example of a connectivity matrice
print(G.nodes())
print(G.edges())
A=nx.adjacency_matrix(G)
print(A.todense())

A=A*2    # weight of the edges
print(A.todense())

WIE=nx.complete_graph(512)     #paper matrices
WIE=nx.adjacency_matrix(WIE)
WIE=WIE*W0

WII=WIE
WEI=WIE




#module network
import networkx as nx
G=nx.Graph()
listneurons=list(range(512))
print(listneurons)
G.add_nodes_from(listneurons)
print(G.nodes)
G.add_edges_from([1,2])
G.add_edges_from([(1,2),(1,3)])

A=nx.adjacency_matrix(G, weight='3')
print(A.todense())

#square matrices
rr=np.zeros((5,5))
print(rr)
rr+=1
print(rr)