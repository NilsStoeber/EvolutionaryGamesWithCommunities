# -*- coding: utf-8 -*-
'''
Rate-Of-Change Vektorfeld für Kooperationsspiele in 2 Communities
Update-Regel: Geglättete Imitation mit Sigmoid-Funktion

In Konsistenz mit den anderen Dateien für diese Update-Regel
ist diese Datei ebenfalls nur für 2 Communities und 2 Strategien aufgesetzt.
'''

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import random as r
import seaborn as sb
import time
from sys import exit

#%% Parameter-Initialisierung

# Anzahl Strategien, nicht ändern!
n_strat = 2

# Auszahlungsparameter
par_R = np.array([8,5])  
par_S = np.array([0,0])
par_T = np.array([3,6])
par_P = np.array([3,1])

'''
Option, die Nash-Gleichgewichte pro Community darzustellen
Game-Types:
 - 'SH' Stag Hunt/Hirschjagd,                   R > T = P > S
 - 'SD' SnowDrift                               T > R > S > P
 - 'PD' Prisoners Dilemma/Gefangenendilemma     T > R > P > S
'''
plot_nasheq = True
game_type = ['SH', 'PD']

n = np.size(par_R)

# Zufallsparameter (Niedrig = Niedriger Zufallsfaktor, Hoch = Hoher Zufallsfaktor)
K_randomness = 1
random_drift = 0.1

# Größen der Population und Communities
N = 200
N_i = np.repeat(N//n, n)
#N_i = np.array([100,150])       #Option Custom Populationseinstellung
N = np.sum(N_i)
states = np.prod(N_i+1)

if n != N_i.size:
    print("FEHLER: Bitte die Anzahl von Auszahlungsparametern und Communities gleich wählen!")
    exit()
    
# Interaktionsparameter
param_inter = 0.05
param_intra = 0.5
par_lambda = param_inter * np.ones((n,n)) + (param_intra-param_inter) * np.eye(n)   #Matrix der Interaktionsparameter
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
def F(y,x):
    # y = alte Strategie, x = potentielle neue Strategie
    return 1/(1 + np.e**(-(x-y)/K_randomness))

# Auszahlung der reinen Strategien gegen die aktuelle Population
def updatepi(f):     
    return 1/N*np.eye(n_strat*n)@A_star@f

# Wahrscheinlichkeit, dass jemand aus Community i (=k//2) und Strategie C (k%2==0) bzw D (k%2==1) zur anderen Strategie wechselt
def p(f,pi,k):
    new_strat = 1 - (k%n_strat)
    prob = 0
    for j in range(n):
        l = n_strat*j+new_strat
        prob += par_lambda[k//n_strat,j]*f[k]*f[l]/param_total*F(pi[k],pi[l])
    return prob

# Übergangsrate, dass jemand aus Community i (=k//2) und Strategie C (k%2==0) bzw D (k%2==1) zur anderen Strategie wechselt
def q(f,pi,k):
    return param_total * p(f,pi,k) + 1/n_strat*random_drift*N_i.T@par_lambda[k//n_strat]*f[k]

# Übergangsraten für den Zustand f
def createQ(f):
    rates = np.zeros((n_strat*(n_strat-1)*n+1))
    pi = updatepi(f)
    for i in range(n_strat*(n_strat-1)*n):
        rates[i] = q(f,pi,i)
    rates[-1] = -sum(rates)
    return rates

# Übergangsraten für alle Zustände
def createQFull():
    Q = np.zeros((states, (n_strat*(n_strat-1)*n+1)))
    for j in range(states):
        k = j//(N_i[1]+1)
        l = j%(N_i[1]+1)
        f = np.array([k,N_i[0]-k,l,N_i[1]-l])
        Q[j,:] = createQ(f)
    return Q

# Bereinigte Übergangswahrscheinlichkeiten für den Zustand f
def createPi(f):
    probs = createQ(f)
    if abs(probs[-1]) > 10e-10:
        probs = probs/probs[-1]
    return probs

# Bereinigte Übergangswahrscheinlichkeiten für alle Zustände
def createPiFull():
    P = np.zeros((states, n_strat*(n_strat-1)*n+1))
    for j in range(states):
        k = j//(N_i[1]+1)
        l = j%(N_i[1]+1)
        f = np.array([k,N_i[0]-k,l,N_i[1]-l])
        P[j,:] = createPi(f)
    return P

#%%

def plotVF(vecscale = 3.0, plot_NE = plot_nasheq):
    P = createPiFull()
    Q = createQFull()
    
    # Verweilparameter
    Qdiag = -Q[:,-1].reshape((N_i[0]+1,N_i[1]+1))
    # Effektive Veränderungsrate (Verweilparameter * 1-Norm des nächsten Übergangs)
    Qdirec = np.copy(Qdiag)
    
    x,y = np.meshgrid(np.linspace(0.5,N_i[1]+0.5,N_i[1]+1),np.linspace(0.5,N_i[0]+0.5,N_i[0]+1))
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    for j in range(states):
        k = j//(N_i[1]+1)
        l = j%(N_i[1]+1)
        dy = -P[j,1] + P[j,0]
        dx = -P[j,3] + P[j,2]
        Qdirec[k,l] = Qdiag[k,l]*(abs(dy)+abs(dx))
        u[N_i[0]-k,l] = dx
        v[N_i[0]-k,l] = dy
    
    sb.heatmap(np.flip(Qdirec, axis = 0), cmap = "Spectral_r", yticklabels = False, xticklabels = False)
    
    #Nash-Gleichgewicht Hirschjagd pro Community
    if plot_NE == True:
        match game_type[0]:
            case 'SH':
                plt.hlines((1-par_T[0]/par_R[0])*N_i[0], 0, N_i[1], color = 'black')
            case 'SD':
                plt.hlines((1-par_S[0]/par_R[0])*N_i[0], 0, N_i[1], color = 'black')
            case 'PD':
                plt.hlines(N_i[0], 0, N_i[1], color = 'black')
        match game_type[1]:
            case 'SH':
                plt.vlines(par_T[1]/par_R[1]*N_i[1], 0, N_i[0], color = 'black')
            case 'SD':
                plt.vlines(par_S[1]/par_R[1]*N_i[1], 0, N_i[0], color = 'black')
            case 'PD':
                plt.vlines(0, 0, N_i[0], color = 'black')
        
    
    plt.axis('off')
    plt.axis('equal')
    vf_steps = N_i//15
    plt.quiver(x[::vf_steps[0],::vf_steps[1]],y[::vf_steps[0],::vf_steps[1]],u[::vf_steps[0],::vf_steps[1]],v[::vf_steps[0],::vf_steps[1]])
    plt.yticks()
    plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
    plt.text(-0.04*N_i[1], 1.04*N_i[0], "D", va = 'top', ha = 'center')
    plt.text(0.99*N_i[1],1.04*N_i[0],"C", va = 'top', ha = 'center')
    plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
    plt.text(0.5*N_i[1], 1.04*N_i[0], "Community 2", va = 'top', ha = 'center')
    plt.title("Erwartete Strategienwechsel pro Zeiteinheit")
    
    plt.show()



plotVF(plot_NE = plot_nasheq)