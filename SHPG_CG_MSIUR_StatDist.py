# -*- coding: utf-8 -*-
'''
Berechnung der stationären Verteilung für Kooperationsspiele in mehreren Communities
Update-Regel: Geglättete Imitation mit Sigmoid-Funktion mit zufälligem Drift
'''

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse as spar
import seaborn as sb
import time
from sys import exit

#%% Parameter-Initialisierung

# Anzahl Strategien, nicht ändern!
n_strat = 2

# Auszahlungsparameter
par_R = np.array([6,6])  
par_S = np.array([0,0])
par_T = np.array([3,3])
par_P = np.array([3,3])
n = np.size(par_R)

# Zufallsparameter (Niedrig = Niedriger Zufallsfaktor, Hoch = Hoher Zufallsfaktor)
K_randomness = 1.0
random_drift = 0.1

# Größen der Population und Communities
N = 200
N_i = np.repeat(N//n, n)
#N_i = np.array([50,50,25])       #Option Custom Populationseinstellung
N = np.sum(N_i)
states = np.prod(N_i+1)

if n != N_i.size:
    print("FEHLER: Bitte die Anzahl von Auszahlungsparametern und Communities gleich wählen!")
    exit()
    
# Interaktionsparameter
param_inter = 0.05
param_intra = 0.5
par_lambda = param_inter * np.ones((n,n)) + (param_intra-param_inter) * np.eye(n)   #Matrix der Interaktionsparameter
# Option: Custom Lambda-Matrix
#par_lambda = np.array([[0.5,0.05,0.1],[0.05,0.5,0.1],[0.1,0.1,0.5]])
param_total = N_i.T @ par_lambda @ N_i

if par_lambda.shape != (n,n):
    print("FEHLER: Bitte die Matrix der Interaktionsparameter passend zur Anzahl der Communities wählen!")
    exit()
    
    
A = np.array([[par_R, par_S], [par_T, par_P]])
A = np.transpose(A, (2,0,1))                                                        #A[i] ist die Auszahlungsmatrix von Community i
A_star = np.tile(A, n).reshape((n_strat*n,n_strat*n)) * par_lambda.repeat(2,0).repeat(2,1)      #Erweiterte und gewichtete Auszahlungsmatrix

if A.shape != (n,n_strat,n_strat):
    print("FEHLER: Bitte die Anzahl von Strategien in diesem Programm bei 2 belassen!")
    exit()




#%% Hilfsfunktionen

# Entscheidungsfunktion: Sigmoid-Funktion
def F(y,x): # y alt, x neu
    return 1/(1 + np.e**(-(x-y)/K_randomness))

# Auszahlung der reinen Strategien gegen die aktuelle Population
def updatepi(f):     
    return 1/N*np.eye(n_strat*n)@A_star@f

# Wahrscheinlichkeit, dass jemand aus Community i (=k//2) und Strategie C (k%2==0) bzw D (k%2==1) zur anderen Strategie wechselt
def p(f,pi,k):
    new_strat = 1 - (k%n_strat)
    prob = 0
    for j in range(n):
        if j == k//n_strat:
            param = param_intra
        else: param = param_inter
        l = n_strat*j+new_strat
        prob += param*f[k]*f[l]/param_total*F(pi[k],pi[l])
    return prob

# Übergangsrate, dass jemand aus Community i (=k//2) und Strategie C (k%2==0) bzw D (k%2==1) zur anderen Strategie wechselt
def q(f,pi,k):
    return 1/n_strat*random_drift*N_i.T@par_lambda[k//n_strat]*f[k] + param_total * p(f,pi,k)


#%% Start der Eigenvektorberechnung

# Erstellung der Übergangsmatrix
states = np.prod(N_i+1)
Pdata = np.zeros((states,2*n+1))
Qdata = np.zeros((states,2*n+1))
offset = np.hstack((0,1,np.cumprod(N_i[n:0:-1]+1), -1, -np.cumprod(N_i[n:0:-1]+1)))
# 0, 1, N2+1, (N1+1)*(N2+1), -1, ...

# 1: Q-Matrix
# 2: Unbereinigte Übergangsmatrix
variant = 1

start_time = time.time()

for k in range(states):
    state = np.zeros(n)
    state[0] = k//offset[n]
    for c in range(1,n):
        state[c] = (k - np.sum((state*offset[n:0:-1])[:c]))//offset[n-c]
    f = np.zeros(2*n)
    f[::2] = state
    f[1::2] = N_i - state

    pi = updatepi(f)

    match variant:
        case 1:
            for c in range(n):
                if state[c] != 0:               # C zu D in Community c  
                    Qdata[k,2*n-c] = q(f,pi,2*c)
                if state[c] != N_i[c]:          # D zu C in Community c  
                    Qdata[k,n-c] = q(f,pi,2*c+1)
            
            Qdata[k,0] = -sum(Qdata[k,:])
        case 2:
            for c in range(n):
                if state[c] != 0:               # C zu D in Community c  
                    Pdata[k,2*n-c] = p(f,pi,2*c)
                if state[c] != N_i[c]:          # D zu C in Community c  
                    Pdata[k,n-c] = p(f,pi,2*c+1)
              
            Pdata[k,0] = 1 - sum(Pdata[k,:])    # Wahrscheinlichkeit Empty-Step

for c in range(n):
    Qdata[:offset[n+1+c],n+1+c] = Qdata[-offset[n+1+c]:,n+1+c]
    Pdata[:offset[n+1+c],n+1+c] = Pdata[-offset[n+1+c]:,n+1+c]

Q = spar.diags_array(Qdata.T, offsets = offset)
P = spar.diags_array(Pdata.T, offsets = offset)

match variant:
    case 1:
        eigval, eigvec = spar.linalg.eigs(Q.T, sigma = 0, k=1)
        w1 = np.real(eigvec).T[0]
        w1 = w1/sum(w1)
        dist = w1.reshape(N_i+1)
        # Bei 2-dimensionaler Randverteilung: Flip bei erster Dimension, die angezeigt wird!

    case 2:
        eigval, eigvec = spar.linalg.eigs(P.T, sigma = 1, k=1)
        w2 = np.real(eigvec).T[0]
        w2 = w2/sum(w2)
        
        dist = w2.reshape(N_i+1)
        
end_time = time.time()
print(variant, end_time - start_time)
#%% Heatmaps

if n==2:
    sb.heatmap(np.flip(dist,0), cmap = "Spectral", yticklabels = False, xticklabels = False)
    #plt.title(f'Absorptionswahrscheinlichkeit für Kooperation für P={par_P},R={par_R}, S={par_S}, T={par_T}, lambda_ii = {param_intra}, lambda_ij = {param_inter}', pad = 20)
    plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
    plt.text(-0.04*N_i[1], 1.04*N_i[0], "D", va = 'center', ha = 'center')
    plt.text(0.99*N_i[1],1.04*N_i[0],"C", va = 'center', ha = 'center')
    plt.text(-0.04*N_i[1], 0.5*N_i[0], "Community 1", rotation = 90, ha = 'center', va = 'center')
    plt.text(0.5*N_i[1], 1.04*N_i[0], "Community 2", va = 'center', ha = 'center')
    plt.show()     
    
if n==3:
    # Zweidimensionale Randverteilung bedingt auf N_3C = i
    for i in range(N_i[2]):
        sb.heatmap(np.flip(dist[:,:,i],0), cmap = "Spectral", yticklabels = False, xticklabels = False)
        #plt.title(f'Absorptionswahrscheinlichkeit für Kooperation für P={par_P},R={par_R}, S={par_S}, T={par_T}, lambda_ii = {param_intra}, lambda_ij = {param_inter}', pad = 20)
        plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
        plt.text(-0.04*N_i[1], 1.04*N_i[0], "D", va = 'center', ha = 'center')
        plt.text(0.99*N_i[1],1.04*N_i[0],"C", va = 'center', ha = 'center')
        plt.text(-0.04*N_i[1], 0.5*N_i[0], "Community 1", rotation = 90, ha = 'center', va = 'center')
        plt.text(0.5*N_i[1], 1.04*N_i[0], "Community 2", va = 'center', ha = 'center')
        plt.show() 
        
    # Zweidimensionale Randverteilung
    dist01 = np.sum(dist, 2)
    sb.heatmap(np.flip(dist01,0), cmap = "Spectral", yticklabels = False, xticklabels = False)
    #plt.title(f'Absorptionswahrscheinlichkeit für Kooperation für P={par_P},R={par_R}, S={par_S}, T={par_T}, lambda_ii = {param_intra}, lambda_ij = {param_inter}', pad = 20)
    plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
    plt.text(-0.04*N_i[1], 1.04*N_i[0], "D", va = 'center', ha = 'center')
    plt.text(0.99*N_i[1],1.04*N_i[0],"C", va = 'center', ha = 'center')
    plt.text(-0.04*N_i[1], 0.5*N_i[0], "Community 1", rotation = 90, ha = 'center', va = 'center')
    plt.text(0.5*N_i[1], 1.04*N_i[0], "Community 2", va = 'center', ha = 'center')
    plt.show()  


#%% Dichten und Verteilungsfunktionen
    
if n==1:
    plt.plot(np.linspace(0,1,N+1), dist)
    plt.xlabel('Anteil Kooperierende')
    plt.ylabel('Dichte', rotation = 90)
    plt.legend(['C in Comm 1', 'C in Comm 2', 'C in Comm 1&2'])
    plt.show()

    
if n==2:
    dist1 = np.sum(dist,1)
    dist2 = np.sum(dist,0)
    dist12 = np.zeros(N+1)
    for i in range(N+1):
        dist12[i] = np.trace(np.flip(dist,0),offset = -N_i[0]+i)
    plt.plot(np.linspace(0,1,N_i[0]+1),dist1*(N_i[0]+1))
    plt.plot(np.linspace(0,1,N_i[1]+1),dist2*(N_i[1]+1))
    plt.plot(np.linspace(0,1,N+1),dist12*(N+1))
    plt.xlabel('Anteil Kooperierende')
    plt.ylabel('Dichte', rotation = 90)
    plt.legend(['C in Comm 1', 'C in Comm 2', 'C in Comm 1&2'])
    #plt.title('Randverteilungen (Zähldichte) der Communities')
    plt.show()
    
    plt.plot(np.linspace(0,1,N_i[0]+1),np.cumsum(dist1))
    plt.plot(np.linspace(0,1,N_i[1]+1),np.cumsum(dist2))
    plt.plot(np.linspace(0,1,N+1),np.cumsum(dist12))
    plt.legend(['C in Comm 1', 'C in Comm 2', 'C in Comm 1&2'])
    #plt.title('Randverteilungen (Zähldichte) der Communities')
    plt.show()
    
if n==3:
    dist1 = np.sum(np.sum(dist,2),1)
    dist2 = np.sum(np.sum(dist,2),0)
    dist3 = np.sum(np.sum(dist,1),0)
    dist12 = np.zeros(np.sum(N_i[:2])+1)
    for i in range(np.sum(N_i[:2])+1):
        dist12[i] = np.trace(np.flip(dist01,0),offset = -N_i[0]+i)
    plt.plot(np.linspace(0,1,N_i[0]+1),dist1*(N_i[0]+1))
    plt.plot(np.linspace(0,1,N_i[1]+1),dist2*(N_i[1]+1))
    plt.plot(np.linspace(0,1,N_i[2]+1),dist3*(N_i[2]+1))
    plt.plot(np.linspace(0,1,np.sum(N_i[:2])+1), dist12*(np.sum(N_i[:2])+1))
    plt.legend(['C in Comm 1', 'C in Comm 2', 'C in Comm 3', 'C in Comm 1&2'])
    #plt.title('Randverteilungen (Zähldichte) der Communities')
    plt.show()
    
    plt.plot(np.linspace(0,1,N_i[0]+1),np.cumsum(dist1))
    plt.plot(np.linspace(0,1,N_i[1]+1),np.cumsum(dist2))
    #plt.plot(np.linspace(0,1,N_i[2]+1),np.cumsum(dist3))
    plt.plot(np.linspace(0,1,np.sum(N_i[:2])+1), np.cumsum(dist12))
    plt.legend(['C in Comm 1', 'C in Comm 2', 'C in Comm 1&2'])
    #plt.title('Randverteilung (Verteilungsfunktion) der Communities')
    plt.show()
    

