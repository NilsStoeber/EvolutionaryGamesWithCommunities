# -*- coding: utf-8 -*-
'''
Monte-Carlo-Simulation für Kooperationsspiele im Stochastischen Blockmodell
Update-Regel: Logit-Regel mit Boltzmann-Funktion

Um die GrößŸe des Zustandsraumes in angemessenen GrößŸen
zu lassen, ist die Simulation in dieser Datei lediglich für 2 Communities und 2 Strategien aufgesetzt.
'''

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import random as r
import seaborn as sb
import time
import networkx as nx
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

# GröŸen der Population und Communities
N = 100
N_i = np.repeat(N//n, n)
# Option: Custom Populationseinstellung
#N_i = np.array([100,50])       
N = np.sum(N_i)
c_jumps = np.cumsum(N_i).astype(int)

if n != N_i.size:
    print("FEHLER: Bitte die Anzahl von Auszahlungsparametern und Communities gleich wählen!")
    exit()
    
# Interaktionsparameter
param_inter = 0.05
param_intra = 0.5
par_lambda = param_inter * np.ones((n,n)) + (param_intra-param_inter) * np.eye(n)   #Matrix der Interaktionsparameter
param_total = N_i.T @ par_lambda @ N_i
# Option: Custom Lambda-Matrix
#par_lambda = np.array([[0.5,0.05],[0.05,0.5]])

if par_lambda.shape != (n,n):
    print("FEHLER: Bitte die Matrix der Interaktionsparameter passend zur Anzahl der Communities wählen!")
    exit()
    
A_pay = np.array([[par_R, par_S], [par_T, par_P]])
A_pay = np.transpose(A_pay, (2,0,1))

if A_pay.shape != (n,n_strat,n_strat):
    print("FEHLER: Bitte die Anzahl von Strategien in diesem Programm bei 2 belassen!")
    exit()
    
maxDur = 10.0
seed = 1

fStart = np.ones((n)).astype(int)

#%% Hilfsfunktionen

# Zufällige Startverteilung über Communities hinweg
def randomStrategies(f):
    strategies = np.repeat(np.linspace(1,n_strat,n_strat), f)
    strategies = np.random.permutation(strategies)
    return strategies.astype(int)

# Zufällige Startverteilung pro Community
def randomStrategiesByComm(fByComm):
    strategies = np.zeros(N, dtype = int)
    for i in range(n):
        if i == 0:
            strategies[:c_jumps[0]] = randomStrategies(np.hstack((fByComm[i],N_i[i] - fByComm[i])))
        else:
            strategies[c_jumps[i-1]:c_jumps[i]] = randomStrategies([fByComm[i],N_i[i] - fByComm[i]])
    return strategies.astype(int)

# Logit-Regel
def F(new_strat,pi):
    return np.exp(pi[new_strat]/K_randomness) / np.sum(np.exp(pi/K_randomness))

# Berechnung der Auszahlung eines Spielers mit Strategie "strat" in Community "comm" gegen die Strategien seiner Nachbarn "strat_adj"
def updatepi(strat, comm,strat_adj):  
    return np.squeeze(1/N*np.eye(n_strat)[strat-1]@A_pay[comm]@strat_adj.T)

# Erstellung eines SBM-Graphen (mit Nebenbedingung einer Kommunikationsklasse)
def SBMGraph(N_i,par_lambda, draw = True):
    n_cc = False
    for i in range(100):
        # Connected Component Check
        if n_cc == True: break
        G = nx.stochastic_block_model(N_i, par_lambda)

        n_cc = nx.is_connected(G)
        A_adj = (nx.adjacency_matrix(G)).todense()
    
    if draw == True:
        nx.draw(G, with_labels = True)
        plt.show()
        
    if n_cc == True: 
        return A_adj, G
    else: return 0, 0

# Simuliere einen Strategienwechsel
def stratChange(G, A_adj,f, strategies):
    player = r.sample(population_sample, k=1)
    comm = np.argmax(np.cumsum(N_i) > player)
    current_strat = strategies[player]
    strategies_adjacent = np.apply_along_axis(np.bincount, 1, A_adj*strategies, minlength = n_strat+1)[player,1:]
    payouts = np.zeros(n_strat)
    P_change = np.zeros(n_strat)
    for i in range(n_strat):
        payouts[i] = updatepi(i+1,comm, strategies_adjacent)
    for i in range(n_strat):
        P_change[i] = F(i,payouts)
    
    randVal = np.random.uniform()
    new_strat = np.argmax(np.cumsum(P_change) > randVal)+1

    f[n_strat*comm + current_strat-1] -= 1
    f[n_strat*comm + new_strat-1] += 1
    strategies[player] = new_strat

    return f, strategies

#%% 

r.seed(seed)
np.random.seed(seed)

A_adj, G = SBMGraph(N_i, par_lambda)
neighbor_count = np.sum(A_adj, axis = 1)
population = np.linspace(0, N-1, N, dtype = int)
population_sample = list(np.repeat(population, neighbor_count))
edges = list(G.edges)

#%% Simulation

def simulation(G, N = N, f_init = fStart, StartByComm = True, timeInterval = 0.1, maxduration = 5.0):
    
    A_adj = (nx.adjacency_matrix(G)).todense()
    par_total = np.sum(A_adj)
    
    duration = 0
    trackerTime = [0]
    tracker1 = [f_init[0]]
    tracker2 = [f_init[1]]
    trackerState = np.zeros((N_i[0]+1,N_i[1]+1))
    
    f = np.zeros((n_strat*n), dtype = int)
    f[::n_strat] = f_init
    f[1::n_strat] = N_i - f_init
    match StartByComm:
        case True:
            strategies = randomStrategiesByComm(f_init)
        case False:
            strategies = randomStrategies(f_init)
      
    f[:2] = np.bincount(strategies[:c_jumps[0]],minlength = n_strat+1)[1:]
    for i in range(1,n):
        f[2*i:2*(i+1)] = np.bincount(strategies[c_jumps[i-1]:c_jumps[i]],minlength = n_strat+1)[1:]
    
    while duration < maxduration:
        t = np.random.exponential(1/par_total)
        duration_old = duration
        duration+=t
        
        f, strategies = stratChange(G, A_adj, f, strategies)
        trackerTime.append(duration)
        tracker1.append(f[0])
        tracker2.append(f[2])
        
        if (duration//timeInterval) != (duration_old//timeInterval):
            trackerState[f[0],f[2]] += (duration//timeInterval) - (duration_old//timeInterval)
    
    return(trackerTime,tracker1,tracker2, trackerState)


#%% Zeige Verlauf eines Sample-Pfades
times, l1,l2,l12 = simulation(G, N=N, f_init = fStart)
plt.plot(times,l1)
plt.plot(times,l2)
plt.show()


#%% Start der Simulation
start_time = time.time()
B = np.zeros((N_i[0]+1,N_i[1]+1))
for i in range(N_i[0]+1):
    for j in range(N_i[1]+1):
        B += simulation(G, f_init = [i,j], maxduration = maxDur)[-1]
        print(i,j)
B = np.flip(B, 0)/np.sum(B)

end_time = time.time()


#%% Darstellung der stationären Verteilung + Laufzeitinformationen

sb.heatmap(B, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

print()
print('Gesamtzeit: ', int((end_time - start_time)//60),'Minuten, ', int((end_time - start_time)%60), 'Sekunden.')
print('MC-Simulationen der Länge: ', maxDur, ' bei ', np.prod(N_i+1), ' Zuständen')
print('Gesamtpopulation N: ', sum(N_i))


#%% Darstellung der Dichten und Verteilungsfunktionen (1- und 2-dimensional)

dist = np.flip(B,0)
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
    plt.title('Randverteilungen (Zähldichte) der Communities')
    plt.show()
    
    plt.plot(np.linspace(0,1,N_i[0]+1),np.cumsum(dist1))
    plt.plot(np.linspace(0,1,N_i[1]+1),np.cumsum(dist2))
    plt.plot(np.linspace(0,1,N+1),np.cumsum(dist12))
    plt.legend(['C in Comm 1', 'C in Comm 2', 'C in Comm 1&2'])
    plt.title('Randverteilungen (Verteilungsfunktion) der Communities')
    plt.show()