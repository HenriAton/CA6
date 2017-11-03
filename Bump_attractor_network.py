import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import networkx as nx              # module useful to manipulate graphs such as connectivity matrices

## Define the parameters

# All-to-all connectivity matrices
W0=1  #not given in the paper

GEE=6	 	# strength of excitation to excitatory neurons
GEI=4		# strength of excitation to inhibitory neurons
GIE=3.4
GII=0.85

# Input-output function
Im=0
Ie0par=0.5
Ii0par=0.2
re0=               #to complete
ri0=               #to complete
sigmae=1
sigmai=3
taue=20
taui=10

## Build the network

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

WEE=                       #circular Gaussian function of i – j        !!!

Ie0=(np.zeros((512,512))+1)*Ie0par   #initial current for excitatory neurons
Ii0=(np.zeros((512,512))+1)*Ii0par   #initial current for inhibitory neurons

# Input-output function

Ie=GEE*WEE-GEI*WEI+Ie0+Im
Ii=GIE*WIE-GII*WII+Ii0

def transfer(x):
    if x>0 and x<1:
        x=x*x
    if x>=1:
        x=np.sqrt(4*x-3)   ## output must be matrix or single value?  !!!
    
phie=transfer(Ie) 
phii=transfer(Ii)

# Firing-rate
re=(re0-phie-sigmae*epsilon0)*np.exp(-t/taue)+phie+sigmae*epsilon

ri=(ri0-phii-sigmai*epsilon0)*np.exp(-t/taui)+phii+sigmai*epsilon






## Draft

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