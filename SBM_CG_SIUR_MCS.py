# -*- coding: utf-8 -*-
'''
Monte-Carlo-Simulation für Kooperationsspiele im Stochastischen Blockmodell
Update-Regel: Geglättete Imitation

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

# GröŸen der Population und Communities
N = 50
N_i = np.repeat(N//n, n)
# Option: Custom Populationseinstellung
#N_i = np.array([100,50])       
N = np.sum(N_i)
c_jumps = np.cumsum(N_i).astype(int)

if n != N_i.size:
    print("FEHLER: Bitte die Anzahl von Auszahlungsparametern und Communities gleich wählen!")
    exit()
    
# Interaktionsparameter
param_inter = 0.05 #p_ij
param_intra = 0.5  #p_ii
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
    
tSpot = 100
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
    strategies = np.zeros(N)
    c_jumps = np.cumsum(N_i).astype(int)
    for i in range(n):
        if i == 0:
            strategies[:c_jumps[0]] = randomStrategies(np.hstack((fByComm[i],N_i[i] - fByComm[i])))
        else:
            strategies[c_jumps[i-1]:c_jumps[i]] = randomStrategies([fByComm[i],N_i[i] - fByComm[i]])
    return strategies.astype(int)

# Sigmoid-Funktion für Geglättete Imitation
def F(y,x): # y alt, x neu
    return 1/(1 + np.e**(-(x-y)/K_randomness))

# Berechnung der Auszahlung eines Spielers mit Strategie "strat" in Community "comm" gegen die Strategien seiner Nachbarn "strat_adj"
def updatepi(strat, comm,strat_adj):     
    return 1*np.eye(n_strat)[strat-1]@A_pay[comm]@strat_adj

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

# Berechne Auszahlung für alle gesuchten Spieler
def randomPayout(A_adj,strategies, players = range(N)):
    strategies_adjacent = np.apply_along_axis(np.bincount, 1, A_adj*strategies, minlength = n_strat+1)[:,1:]
    payouts = np.zeros(N)
    for i in players:
        strat = strategies[i]
        comm = np.argmax(np.cumsum(N_i) > i)
        payouts[i] = updatepi(strat,comm,strategies_adjacent[i])
    return payouts

def stratChange(G, A_adj,f, strategies):
    edge_num = np.random.randint(0,len(edges))
    edge = edges[edge_num]
    i,j = edge[0],edge[1]
    
    if strategies[i] == strategies[j]:
        return f, strategies
    payouts = randomPayout(A_adj, strategies, [i,j])
    
    randVal = np.random.random()
    if randVal < F(payouts[i],payouts[j]):
        f[strategies[i]-1] -= 1
        f[strategies[j]-1] += 1
        strategies[i] = strategies[j]
    else: 
        f[strategies[j]-1] -= 1
        f[strategies[i]-1] += 1
        strategies[j] = strategies[i]
        
    return f, strategies

#%%

r.seed(seed)
np.random.seed(seed)

A_adj, G = SBMGraph(N_i, par_lambda)
edges = list(G.edges)

#%% Simulation

def simulation(G, N = N, f_init = fStart, StartByComm = True, maxduration = 5.0):
    
    A_adj = (nx.adjacency_matrix(G)).todense()
    par_total = np.sum(A_adj)
    
    duration = 0
    trackerTime = [0]
    tracker1 = [f_init[0]]
    tracker2 = [f_init[1]]
    
    f = f_init
    match StartByComm:
        case True:
            strategies = randomStrategiesByComm(f)
        case False:
            strategies = randomStrategies(f)
    f = np.bincount(strategies,minlength = n_strat+1)[1:]
    

    while max(f[0],N-f[0]) < N and duration < maxduration:
        t = np.random.exponential(1/par_total)
        duration+=t
        
        f, strategies = stratChange(G, A_adj, f, strategies)
        trackerTime.append(duration)
        tracker1.append(f[0])
        tracker2.append(f[1])
        
    return(f[0], tracker1, tracker2, trackerTime)

def simulateSpot(G, N, fStart, tSpot):
    C = 0
    D = 0
    timeC = 0
    timeD = 0
    timeGen = 0
    # Zähle Absorptionen und die zugehÃ¶rigen Zeiten pro Strategie
    for k in range(tSpot):
        res = simulation(G,N,fStart,StartByComm = True)
        if res[0] == N:
            C += 1
            timeC += res[3][-1]
            timeGen += res[3][-1]
        elif res[0] == 0:
            D += 1
            timeD += res[3][-1]
            timeGen += res[3][-1]

    # Durchschnittsbildung der Absorptionszeiten
    if C != 0:
        timeC = timeC/C
    else: timeP = 'nan'
    if D != 0:
        timeD = timeD/D
    else: timeS = 'nan'
    if (C+D) != 0:
        timeGen = timeGen/(C+D)
    else: timeGen = 'nan'
    return (C/tSpot,D/tSpot,timeC,timeD, timeGen)

#%% Zeige Verlauf eines Sample-Pfades

fC, l1,l2,times = simulation(G, N=N, f_init = fStart)
plt.plot(times[1:],l1[1:])
plt.plot(times[1:],l2[1:])
plt.show()

    
#%% Start der Simulation

start_time = time.time()
pC = np.zeros(N_i+1)
pD = np.zeros(N_i+1)
tC = np.zeros(N_i+1)
tD = np.zeros(N_i+1)
tGen = np.zeros(N_i+1)

states = np.prod(N_i+1)

# Optional für Fortschrittsanzeige

#print("[" + " "*100 + "]")
#print(" ", end = "")


offset = np.hstack((0,1,np.cumprod(N_i[n:0:-1]+1), -1, -np.cumprod(N_i[n:0:-1]+1)))

for k in range(states):
    state = np.zeros(n)
    state[0] = k//offset[n]
    for c in range(1,n):
        state[c] = (k - np.sum((state*offset[n:0:-1])[:c]))//offset[n-c]
    f = np.zeros(2*n)
    f[::2] = state
    f[1::2] = N_i - state
    f = f.astype(int)
    i, j = f[::2]
    
    spot_res = simulateSpot(G,N,[i,j],tSpot)
    pC[i,j] = spot_res[0]
    pD[i,j] = spot_res[1]
    tC[i,j] = spot_res[2]
    tD[i,j] = spot_res[3]
    tGen[i,j] = spot_res[4]
    
    if (pC[i,j] + pD[i,j]) > 0:
        pC[i,j] = pC[i,j] / (pC[i,j] + pD[i,j])
        pD[i,j] = pD[i,j] / (pC[i,j] + pD[i,j])
        
    # Optional für Fortschrittsanzeige
    
    #if k//(states/100) > (k-1)//(states/100):
    #        print("X", end = "")
    
end_time = time.time()

#%% Approx. Absorptionswahrscheinlichkeiten von Cooperation

# Umordnen
pC = np.flip(pC.reshape((N_i[0]+1,N_i[1]+1)),axis = 0)


sb.heatmap(pC, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%% Approx. Absorptionswahrscheinlichkeiten von Defection (wie oben)

# Umordnen
pD = np.flip(pD.reshape((N_i[0]+1,N_i[1]+1)),axis = 0)


sb.heatmap(pD, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%% Approx. Absorptionszeiten von Cooperation (wie oben)

# Umordnen
tGen = np.flip(tGen.reshape((N_i[0]+1,N_i[1]+1)),axis = 0)


sb.heatmap(tGen, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%% Approx. Absorptionszeiten von Cooperation (wie oben)

# Umordnen
tC = np.flip(tC.reshape((N_i[0]+1,N_i[1]+1)),axis = 0)


sb.heatmap(tC, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%% Approx. Absorptionszeiten von Defection (wie oben)

# Umordnen
tD = np.flip(tD.reshape((N_i[0]+1,N_i[1]+1)),axis = 0)


sb.heatmap(tD, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.07*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.07*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.07*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()


print()
print('Gesamtzeit: ', int((end_time - start_time)//60),'Minuten, ', int((end_time - start_time)%60), 'Sekunden.')
print(tSpot, sum(N_i))