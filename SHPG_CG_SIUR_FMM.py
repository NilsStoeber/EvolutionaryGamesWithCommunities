# -*- coding: utf-8 -*-
'''
Fundamentalmatrix-Methode für Kooperationsspiele in 2 Communities
Update-Regel: Geglättete Imitation mit Sigmoid-Funktion

Um die Größe des Zustandsraumes in angemessenen Größen für die Fundamentalmatrix-Methode
zu lassen, ist diese analytische Methode in dieser Datei lediglich für 2 Communities und 2 Strategien aufgesetzt.
'''

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
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

# Größen der Population und Communities
N = 200
N_i = np.repeat(N//n, n)
#N_i = np.array([20,20])       #Option Custom Populationseinstellung
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
    
#%% Funktionen für die Update-Regel

# Entscheidungsfunktion: Sigmoid-Funktion
def F(y,x):
    # y = alte Strategie, x = potentielle neue Strategie
    return 1/(1 + np.e**(-(x-y)/K_randomness))

# Auszahlung der reinen Strategien gegen die aktuelle Population
def updatepi(f):     
    return 1/N*np.eye(n_strat*(n_strat-1)*n)@A_star@f

# Wahrscheinlichkeit, dass jemand aus Community i (=k//2) und Strategie C (k%2==0) bzw D (k%2==1) zur anderen Strategie wechselt
def p(f,pi,k):
    new_strat = 1 - (k%n_strat)
    prob = 0
    for j in range(n):
        l = n_strat*j+new_strat
        prob += par_lambda[k//n_strat,j]*f[k]*f[l]/param_total*F(pi[k],pi[l])
    return prob

# Übergangsrate, dass jemand aus Community i (=k//2) und Strategie C (k%2==0) bzw D (k%2==1) zur anderen Strategie wechselt
def q(f,pi,k):
    return param_total * p(f,pi,k)


#%%

start_time = time.time()

# Erstellung der Übergangsmatrix
# variant = 1: Unbereinigte Sprungmatrix
# variant = 2: Bereinigte Sprungmatrix
# Variante 2 lediglich für Absorptionswahrscheinlichkeiten nutzen
variant = 1

match variant:
    case 1 | 2:
          P = np.zeros((states,states)) 
    case _:
        print("FEHLER: Bitte für die Übergangsmatrix Variante 1 oder 2 wählen!")
        exit()
        


for i in range(N_i[0]+1):
    for j in range(N_i[1]+1):
        k = i*(N_i[1]+1) + j
        f = np.array([i,N_i[0]-i,j,N_i[1]-j])
        pi = updatepi(f)
        if i != 0:       # C zu D in Community 1
            P[k,k-N_i[1]-1] = p(f,pi,0)
        if i != N_i[0]:       # D zu C in Community 1
            P[k,k+N_i[1]+1] = p(f,pi,1)
            P[k,k+N_i[1]+1] = p(f,pi,1)
        if j != 0:       # C zu D in Community 2
            P[k,k-1] = p(f,pi,2)
            P[k,k-1] = p(f,pi,2)
        if j != N_i[1]:       # D zu C in Community 2
            P[k,k+1] = p(f,pi,3)
            P[k,k+1] = p(f,pi,3)
            
        match variant:
            case 1:
                P[k,k] = 1 - sum(P[k,:])    # Wahrscheinlichkeit Empty-Step
            case 2:
                if sum(P[k,:]) > 10e-10:
                    P[k,:] = P[k,:] / sum(P[k,:])
                else: P[k,k] = 1
        
        

#%% Umformen von P in die kanonische Form

Sa = [0,states-1]
St = np.arange(1,states-1)
I_canonical = np.take(np.take(P, Sa, 0), Sa, 1)
R_canonical = np.take(np.take(P, St, 0), Sa, 1)
Q_canonical = np.take(np.take(P, St, 0), St, 1)
P_canonical = np.vstack((np.hstack((Q_canonical,R_canonical)),np.hstack((np.zeros((2,Q_canonical.shape[0])),I_canonical))))


#%% Fundamentalmatrix und Resultate der Fundamentalmatrix-Methode
Nmat = np.linalg.inv(np.identity(Q_canonical.shape[0])-Q_canonical)
c = np.ones(Q_canonical.shape[0])
t = Nmat@c
B = Nmat@R_canonical


Na = [np.zeros_like(Nmat) for i in range(2)]
for i in range(Nmat.shape[0]):
    for j in range(Nmat.shape[1]):
        if B[i,0] > 0:
            Na[0][i,j] = B[j,0]/B[i,0]*Nmat[i,j]
        else: Na[0][i,j] = 'nan'
        if B[i,1] > 0:
            Na[1][i,j] = B[j,1]/B[i,1]*Nmat[i,j]
        else: Na[1][i,j] = 'nan'
        
ta = [Na[i]@c for i in range(2)]




#%% Absorptionswahrscheinlichkeiten von Kooperation

pC = np.flip((np.hstack((I_canonical[0,1],B[:,1],I_canonical[1,1]))).reshape((N_i[0]+1,N_i[1]+1)),axis = 0)

sb.heatmap(pC, cmap = "Spectral", yticklabels = False, xticklabels = False)
#plt.title(f'Absorptionswahrscheinlichkeit für Kooperation für P={par_P},R={par_R}, S={par_S}, T={par_T}, lambda_ii = {param_intra}, lambda_ij = {param_inter}', pad = 20)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[0], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%%
pD = np.flip((np.hstack((I_canonical[0,0],B[:,0],I_canonical[1,0]))).reshape((N_i[0]+1,N_i[1]+1)),axis = 0)

sb.heatmap(pD, cmap = "Spectral", yticklabels = False, xticklabels = False)
#plt.title(f'Absorptionswahrscheinlichkeit für Defection für P={par_P},R={par_R}, S={par_S}, T={par_T}, lambda_ii = {param_intra}, lambda_ij = {param_inter}', pad = 20)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[0], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()


#%%
tGen = np.flip((np.hstack((0,t,0))).reshape((N_i[0]+1,N_i[1]+1)),axis = 0) / param_total

sb.heatmap(tGen, cmap = "Spectral", yticklabels = False, xticklabels = False)
#plt.title(f'Absorptionswahrscheinlichkeit für Defection für P={par_P},R={par_R}, S={par_S}, T={par_T}, lambda_ii = {param_intra}, lambda_ij = {param_inter}', pad = 20)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%%
tC = np.flip((np.hstack((0,ta[1],0))).reshape((N_i[0]+1,N_i[1]+1)),axis = 0) / param_total

sb.heatmap(tC, cmap = "Spectral", yticklabels = False, xticklabels = False)
#plt.title(f'Absorptionswahrscheinlichkeit für Defection für P={par_P},R={par_R}, S={par_S}, T={par_T}, lambda_ii = {param_intra}, lambda_ij = {param_inter}', pad = 20)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%%
tD = np.flip((np.hstack((0,ta[0],0))).reshape((N_i[0]+1,N_i[1]+1)),axis = 0) / param_total

sb.heatmap(tD, cmap = "Spectral", yticklabels = False, xticklabels = False)
#plt.title(f'Absorptionswahrscheinlichkeit für Defection für P={par_P},R={par_R}, S={par_S}, T={par_T}, lambda_ii = {param_intra}, lambda_ij = {param_inter}', pad = 20)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

print()
print(time.time() - start_time)

