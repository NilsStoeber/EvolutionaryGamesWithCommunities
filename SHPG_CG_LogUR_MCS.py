# -*- coding: utf-8 -*-
'''
Monte-Carlo-Simulation für Kooperationsspiele in 2 Communities
Update-Regel: Logit-Regel mit Boltzmann-Funktion

Um die Größe des Zustandsraumes in angemessenen Größen
zu lassen, ist die Simulation in dieser Datei lediglich für 2 Communities und 2 Strategien aufgesetzt.
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
par_R = np.array([6,6])  
par_S = np.array([0,0])
par_T = np.array([3,3])
par_P = np.array([3,3])
n = np.size(par_R)

# Zufallsparameter (Niedrig = Niedriger Zufallsfaktor, Hoch = Hoher Zufallsfaktor)
K_randomness = 0.35

# Größen der Population und Communities
N = 200
N_i = np.repeat(N//n, n)
#N_i = np.array([100,50])       #Option Custom Populationseinstellung
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
    
maxDur = 10.0
simseed = 1

#%% Hilfsfunktionen

# Logit-Regel
def F(k,j,pi):
    new_strat = 1-(k%n_strat)
    
    return np.exp(pi[n_strat*(k//n_strat) + new_strat]/K_randomness).T/ sum([np.exp(pi[n_strat*(k//n_strat) + i]/K_randomness).T for i in range(n_strat)])

# Auszahlung der reinen Strategien gegen die aktuelle Population
def updatepi(f):     
    return 1/N*np.eye(n_strat*n)@A_star@f

# Wahrscheinlichkeit, dass jemand aus Community i (=k//2) und Strategie C (k%2==0) bzw D (k%2==1) zur anderen Strategie wechselt
def p(f,pi,k):
    i = k//n_strat
    j = n_strat*i + 1-(k%n_strat)
    lambda_i = N_i.T@par_lambda[i]
    
    prob = lambda_i * f[k] / param_total * F(k,j,pi)
    return prob

# Übergangsrate, dass jemand aus Community i (=k//2) und Strategie C (k%2==0) bzw D (k%2==1) zur anderen Strategie wechselt
def q(f,pi,k):
    i = k//n_strat
    j = n_strat*i + 1-(k%n_strat)
    lambda_i = N_i.T@par_lambda[i]
    lambda_i_mod = lambda_i * F(k,j,pi)
    return lambda_i_mod * f[k]

# Erstellt Array mit allen Übergangsraten aus dem Zustand f
def createQ(f):
    rates = np.zeros((n_strat*(n_strat-1)*n+1))
    pi = updatepi(f)
    for i in range(n_strat*(n_strat-1)*n):
        rates[i] = q(f,pi,i)
    rates[-1] = -sum(rates)
    return rates

# Erstellt Array mit allen unbereinigten Übergangswahrscheinlichkeiten aus dem Zustand f
def createP(f):
    probs = np.zeros((n_strat*(n_strat-1)*n+1))
    pi = updatepi(f)
    for i in range(n_strat*(n_strat-1)*n):
        probs[i] = p(f,pi,i)
    probs[-1] = 1 - sum(probs)
    return probs

# Übergangsraten für alle Zustände
def createQFull():
    Q = np.zeros((states, n_strat*(n_strat-1)*n+1))
    for j in range(states):
        k = j//(N_i[1]+1)
        l = j%(N_i[1]+1)
        f = np.array([k,N_i[0]-k,l,N_i[1]-l])
        Q[j,:] = createQ(f)
    return Q

# Unbereinigte Übergangswahrscheinlichkeiten für alle Zustände
def createPFull():
    P = np.zeros((states, n_strat*(n_strat-1)*n+1))
    for j in range(states):
        k = j//(N_i[1]+1)
        l = j%(N_i[1]+1)
        f = np.array([k,N_i[0]-k,l,N_i[1]-l])
        P[j,:] = createP(f)
    return P

#%%
P_all = createPFull()
Q_all = createQFull()
#%%

# Simuliert das Kooperationsspiel ausgehend vom Startzustand fParam bis zur maximalen Dauer
# Speichert den ersten neuen Zustand pro Zeitintervall, um die stationäre Verteilung zu approximieren
def simulation(fParam = [0,0], seed = 0, timeInterval = 0.1, maxduration = 5.0):
    if seed != 0:
        np.random.seed(seed)
    f = np.array([fParam[0],N_i[0]-fParam[0],fParam[1],N_i[1]-fParam[1]])
    
    duration = 0
    trackerTime = [0]
    tracker1 = [f[0]]
    tracker2 = [f[2]]
    pi = updatepi(f)
    avrgPi = [f[:2]@pi[:2].T*N/(N_i[0]*N_i.T@par_lambda[0]), f[2:]@pi[2:].T*N/(N_i[1]*N_i.T@par_lambda[1])]
    trackerPi = [avrgPi]
    trackerState = np.zeros((N_i[0]+1,N_i[1]+1))
    

    # Das Spiel läuft, solange noch mehr als eine Strategie existiert
    while duration < maxduration:
        state = (N_i[1]+1)*f[0] + f[2]
        #P = createP(f)
        #Q = createQ(f)
        P = P_all[state,:]
        Q = Q_all[state,:]
        P_mod = -Q/Q[-1]
        
        P_change = 1-P[-1]
        t = np.random.exponential(-1/Q[-1])
        duration_old = duration
        duration += t
        sample = np.random.uniform()
        # Samplen des Strategiewechsels
        if sample < P_mod[0]:                                             #in C1 C zu D
            f[0] -= 1
            f[1] += 1
        elif sample < sum(P_mod[:2]):                      #in C1 D zu C
            f[0] += 1
            f[1] -= 1
        elif sample < sum(P_mod[:3]):         #in C2 C zu D
            f[2] -= 1   
            f[3] += 1
        elif sample < sum(P_mod[:4]):  #in C2 D zu C
            f[2] += 1
            f[3] -= 1
        trackerTime.append(duration)
        tracker1.append(f[0])
        tracker2.append(f[2])
        
        if (duration//timeInterval) != (duration_old//timeInterval):
            trackerState[f[0],f[2]] += (duration//timeInterval) - (duration_old//timeInterval)
    
    # Return: Sprungzeitpunkte, besuchte Zustände (tracker1 & tracker2), gespeicherte Zustände
    return(trackerTime,tracker1,tracker2, trackerState)

#%% Zeige Verlauf eines Sample-Pfades

times, c1, c2, states = simulation([20,20], seed = simseed)
plt.plot(c2,c1)
plt.xlim((0,N_i[0]))
plt.ylim((0,N_i[1]))
plt.show()
plt.plot(times, np.array(c1)/N_i[0], linewidth = 0.8)
plt.plot(times, np.array(c2)/N_i[1], linewidth = 0.8)
#plt.rcParams["figure.figsize"] = (9,3)
plt.xlabel('t')
plt.ylabel('Anteil Kooperierende', rotation = 90)
plt.ylim((0,1))
plt.show()

#%% Start der Simulation

start_time = time.time()
B = np.zeros((N_i[0]+1,N_i[1]+1))
for i in range(N_i[0]+1):
    for j in range(N_i[1]+1):
        B += simulation([i,j], maxduration = maxDur)[-1]
        print(i,j)
B = np.flip(B, 0)/np.sum(B)

end_time = time.time()

#%% Darstellung der stationären Verteilung + Laufzeitinformationen

sb.heatmap(B, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.04*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.04*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.04*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

print()
print('Gesamtzeit: ', int((end_time - start_time)//60),'Minuten, ', int((end_time - start_time)%60), 'Sekunden.')
print('MC-Simulationen der Länge: ', maxDur, ' bei ', np.prod(N_i+1), ' Zuständen')
print('Gesamtpopulation N: ', sum(N_i))