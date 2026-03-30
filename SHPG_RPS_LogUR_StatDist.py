# -*- coding: utf-8 -*-
'''
Berechnung der stationären Verteilung für Schere-Stein-Papier
Update-Regel: Logit-Regel
'''

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import random as r
import seaborn as sb
import scipy.sparse as spar
import time

#%% Parameter-Initialisierung

# Anzahl Strategien, nicht ändern!
n_strat = 3

# Auszahlungsparameter
par_a1 = np.array([2]) # Loss R
par_a2 = np.array([2]) # Loss P
par_a3 = np.array([2]) # Loss S
par_b1 = np.array([1]) # Win R
par_b2 = np.array([1]) # Win P
par_b3 = np.array([1]) # Win S
n = np.size(par_a1)

# Zufallsparameter (Niedrig = Niedriger Zufallsfaktor, Hoch = Hoher Zufallsfaktor)
K_randomness = 0.05

# Größen der Population und Communities
N = 200
N_i = np.repeat(N//n, n)
#N_i = np.array([14,12])       #Option Custom Populationseinstellung
N = np.sum(N_i)

states = np.prod((N_i+1)*(N_i+2)//2)
states_each = (N_i+1)*(N_i+2)//2

# Interaktionsparameter
param_inter = 0.1
param_intra = 0.5
par_lambda = param_inter * np.ones((n,n)) + (param_intra-param_inter) * np.eye(n)   #Matrix der Interaktionsparameter
# Option: Custom Lambda-Matrix
#par_lambda = np.array([[1.0,0.1,[0.1,1.0]])
param_total = N_i.T @ par_lambda @ N_i

if par_lambda.shape != (n,n):
    print("FEHLER: Bitte die Matrix der Interaktionsparameter passend zur Anzahl der Communities wählen!")
    exit()
    

#Auszahlung zeilenweise für Rock/Paper/Scissors
A = np.array([[np.repeat(0,n), -par_a1, par_b1], [par_b2, np.repeat(0,n), -par_a2],[-par_a3, par_b3, np.repeat(0,n)]])
A = np.transpose(A, (2,0,1))                                                        #A[i] ist die Auszahlungsmatrix von Community i
A_star = np.tile(A, n).reshape((n_strat*n,n_strat*n)) * par_lambda.repeat(n_strat,0).repeat(n_strat,1)      #Erweiterte und gewichtete Auszahlungsmatrix

if A.shape != (n,n_strat,n_strat):
    print("FEHLER: Bitte die Anzahl von Strategien in diesem Programm bei 2 belassen!")
    exit()

#%%
# Funktionen für erwartete Auszahlung und die gegebene Update-Regel
def F(k,j,pi): # k neu (0/1/2), j alt (0 ... n_strat*n-1)
    new_strat = k%n_strat + (j//n_strat)*n_strat
    return par_lambda[j//n_strat]@np.exp(pi[k::n_strat]/K_randomness).T/ sum([(par_lambda[j//n_strat]@np.exp(pi[i::n_strat]/K_randomness).T) for i in range(n_strat)])


def updatepi(f):     
    return 1/N*np.eye(n_strat*n)@A_star@f


# Wahrscheinlichkeit, dass jemand aus Community i (=k//3) und Strategie R/P/S (k%n_strat = 0/1/2)
# zur Strategie j wechselt
def p(f,pi,k,new_strat):
    i = k//n_strat
    lambda_i = N_i.T@par_lambda[i]
    
    prob = lambda_i * f[k] / param_total * F(new_strat,k,pi)
    return prob

def q(f,pi,k,new_strat):
    i = k//n_strat
    lambda_i = N_i.T@par_lambda[i]
    lambda_i_mod = lambda_i * F(new_strat,k,pi)
    return lambda_i_mod * f[k]



#%%
start_time = time.time()

# Erstellung der Übergangsmatrix
data = np.zeros((states,n*n_strat*(n_strat-1) + 1))
coords_x = np.zeros((states,n*n_strat*(n_strat-1)+1))
coords_y = np.zeros((states,n*n_strat*(n_strat-1)+1))
start_time = time.time()

idx = [None for i in range(n)]
for c in range(n):
    idx[c] = np.array([(i)*(i+1)//2 for i in range(N_i[c]+2)])

f = np.zeros((n_strat * n))
for state in range(states):
    k1 = state%((N_i[0]+1)*(N_i[0]+2)//2)
    i1 = N_i[0] - np.argwhere(k1 < idx[0])[0,0] +1
    j1 = k1 - idx[0][N_i[0]-i1]
    f[0:3] = np.array([i1,j1,N_i[0]-i1-j1])
    if n == 2:
        k2 = state//((N_i[0]+1)*(N_i[0]+2)//2)
        i2 = N_i[1] - np.argwhere(k2 < idx[1])[0,0] +1
        j2 = k2 - idx[1][N_i[1]-i2]
        f[3:] = np.array([i2,j2,N_i[1]-i2-j2])
    
    pi = updatepi(f)
    
    
    # Wenn Wechsel weg von R (P bzw. S):
    coords_x[state,:] = state
    coords_y[state,0] = state
    for i in range(n_strat * n):
        if f[i] != 0:
            i_comm = i//n_strat
            match i%n_strat:
                case 0:
                    data[state,2*i+1] = q(f,pi,i,1)
                    data[state,2*i+2] = q(f,pi,i,2)
                    #data[state,2*i+1] = 12 + 100*i_comm
                    #data[state,2*i+2] = 13 + 100*i_comm
                    coords_y[state,2*i+1] = state + (N_i[i_comm]-f[i]+2)*np.prod(states_each[:i_comm])
                    coords_y[state,2*i+2] = state + (N_i[i_comm]-f[i]+1)*np.prod(states_each[:i_comm])
                case 1:
                    data[state,2*i+1] = q(f,pi,i,0)
                    data[state,2*i+2] = q(f,pi,i,2)
                    #data[state,2*i+1] = 21 + 100*i_comm
                    #data[state,2*i+2] = 23 + 100*i_comm
                    coords_y[state,2*i+1] = state - (N_i[i_comm]-f[i-1]+1)*np.prod(states_each[:i_comm])
                    coords_y[state,2*i+2] = state - (1)*np.prod(states_each[:i_comm])
                case 2:
                    data[state,2*i+1] = q(f,pi,i,0)
                    data[state,2*i+2] = q(f,pi,i,1)
                    #data[state,2*i+1] = 31 + 100*i_comm
                    #data[state,2*i+2] = 32 + 100*i_comm
                    coords_y[state,2*i+1] = state - (N_i[i_comm]-f[i-2])*np.prod(states_each[:i_comm])
                    coords_y[state,2*i+2] = state + (1)*np.prod(states_each[:i_comm])

data[:,0] = -np.sum(data,axis = 1)
data = data.flatten()
coords_x = coords_x.flatten()
coords_y = coords_y.flatten()

Q = spar.coo_matrix((data, (coords_x, coords_y)), shape = (states,states))

            
                

#%%

eigval, eigvec = spar.linalg.eigs(Q.T, sigma = 0, k=1)
w1 = np.real(eigvec).T[0]
w1 = w1/sum(w1)
#%%
dist = w1.reshape(states_each, order = 'F')
# Bei 2-dimensionaler Randverteilung: Flip bei erster Dimension, die angezeigt wird!

end_time = time.time()

#%%
def marginal_dist(dist, community):
    marg_dist = dist
    for i in range(n-1,-1,-1):
        if i == community: continue
        print(i)
        marg_dist = np.sum(marg_dist, axis = i)

    ER = np.empty((N_i[community]+1, 2*N_i[community]+2)) * np.nan
    for i in range(N_i[community]+1):
        for j in range(i+1):
            ER[i,N_i[community] + i -2*(j):N_i[community] + i -2*(j-1)] = marg_dist[idx[community][i]+j]


                
    sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
    plt.text(-0.1*N_i[community],1.08*N_i[community],"P")
    plt.text(1.03*N_i[community], -0.03*N_i[community], "R")
    plt.text(2.13*N_i[community],1.08*N_i[community],"S")
    plt.show()

for c in range(n):
    marginal_dist(dist, c)

print(end_time - start_time)







