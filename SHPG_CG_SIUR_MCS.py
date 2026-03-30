# -*- coding: utf-8 -*-
'''
Monte-Carlo-Simulation für Kooperationsspiele in 2 Communities
Update-Regel: Geglättete Imitation mit Sigmoid-Funktion

Um die Größe des Zustandsraumes in angemessenen Größen
zu lassen, ist die Simulation in dieser Datei lediglich für 2 Communities und 2 Strategien aufgesetzt.
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
#N_i = np.array([100,100])       #Option Custom Populationseinstellung
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
A_star = np.tile(A, n).reshape((n_strat*n,n_strat*n)) * par_lambda.repeat(n_strat,0).repeat(n_strat,1)      #Erweiterte und gewichtete Auszahlungsmatrix

if A.shape != (n,n_strat,n_strat):
    print("FEHLER: Bitte die Anzahl von Strategien in diesem Programm bei 2 belassen!")
    exit()
    
simSeed = 42
tSpot = 100
#%% Funktionen für die Update-Regel

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

# Übergangsrate, dass jemand aus Community i (=k//2) und Strategie C (k%2==0) bzw D (k%2==1) zur anderen Strategie wechselt
def q(f,pi,k):
    return param_total * p(f,pi,k)

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
#%% Simulation

def simulation(fParam = [0,0], seed = 0, maxduration = 5.0):
    if seed != 0:
        np.random.seed(seed)
    f = np.array([fParam[0],N_i[0]-fParam[0],fParam[1],N_i[1]-fParam[1]])
    
    duration = 0
    trackerTime = [0]
    tracker1 = [f[0]]
    tracker2 = [f[2]]
    steps = 0
    # Das Spiel läuft, solange noch mehr als eine Strategie existiert
    while f[1]+f[3]>0 and f[0]+f[2] > 0 and duration < maxduration:
        state = (N_i[1]+1)*f[0] + f[2]
        #P = createP(f)
        #Q = createQ(f)
        P = P_all[state,:]
        Q = Q_all[state,:]
        Pi = -Q/Q[-1]
        
        t = np.random.exponential(-1/Q[-1])
        
        duration += t
        sample = np.random.uniform()
        # Samplen des Strategiewechsels
        if sample < Pi[0]:                               #in C1 C zu D
            f[0] -= 1
            f[1] += 1
        elif sample < sum(Pi[:2]):                       #in C1 D zu C
            f[0] += 1
            f[1] -= 1
        elif sample < sum(Pi[:3]):                       #in C2 C zu D
            f[2] -= 1   
            f[3] += 1
        elif sample < sum(Pi[:4]):                       #in C2 D zu C
            f[2] += 1
            f[3] -= 1
        # Ansonsten kein Strategiewechsel, Zustand bleibt gleich
        trackerTime.append(duration)
        tracker1.append(f[0])
        tracker2.append(f[2])
        
    # Return: Anzahl Kooperatoren am Ende, Zeit, Tracker der Kooperatoren pro Community
    return(f[0]+f[2],trackerTime,tracker1,tracker2, steps)

final, times, c1, c2, stepcount = simulation([20,20],simSeed)
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

#%% t-fache Simulation des gegebenen Startzustands
def simulateSpot(f, tSpot):
    c = 0
    d = 0
    timeC = 0
    timeD = 0
    timeGen = 0
    # Zähle Absorptionen und die zugehörigen Zeiten pro Strategie
    for k in range(tSpot):
        sim_res = simulation(f)
        if sim_res[0] == 0:
            d += 1
            timeD += sim_res[1][-1]
            timeGen += sim_res[1][-1]
        else:
            c+= 1
            timeC += sim_res[1][-1]
            timeGen += sim_res[1][-1]
    # Durchschnittsbildung der Absorptionszeiten
    if c != 0:
        timeC = timeC/c
    else: timeC = 'nan'
    if d != 0:
        timeD = timeD/d
    else: timeD = 'nan'
    if (c+d) != 0:
        timeGen = timeGen/(c+d)
    return (c/tSpot,d/tSpot,timeC,timeD,timeGen)

#%% Aproximiere Absorptionswahrscheinlichkeiten und -zeiten für jeden Zustand durch Simulation

start_time = time.time()

pC = np.zeros((N_i[0]+1,N_i[1]+1))
pD = np.zeros((N_i[0]+1,N_i[1]+1))
tC = np.zeros((N_i[0]+1,N_i[1]+1))
tD = np.zeros((N_i[0]+1,N_i[1]+1))
tGen = np.zeros((N_i[0]+1,N_i[1]+1))

total = (N_i[0]+1)*(N_i[1]+1)

# Optional für Fortschrittsanzeige
#print("[" + " "*100 + "]")
#print(" ", end = "")
count = 0
for i in range(N_i[0]+1):
    for j in range(N_i[1]+1):
        spot_res = simulateSpot([i,j],tSpot)
        pC[i,j] = spot_res[0]
        pD[i,j] = spot_res[1]
        tC[i,j] = spot_res[2]
        tD[i,j] = spot_res[3]
        tGen[i,j] = spot_res[4]
        count+=1
        
        # Optional für Fortschrittsanzeige
        #if count//(total/100) > (count-1)//(total/100):
        #    print("X", end = "")
        


#%% Approx. Absorptionswahrscheinlichkeiten von Cooperation

# Umordnen
pC = np.flip(pC.reshape((N_i[0]+1,N_i[1]+1)),axis = 0)

sb.heatmap(pC, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.04*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.04*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.04*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%% Approx. Absorptionswahrscheinlichkeiten von Defection (wie oben)

# Umordnen
pD = np.flip(pD.reshape((N_i[0]+1,N_i[1]+1)),axis = 0)

sb.heatmap(pD, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.04*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.04*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.04*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%% Approx. Absorptionszeiten (wie oben)

# Umordnen
tGen = np.flip(tGen.reshape((N_i[0]+1,N_i[1]+1)),axis = 0)

sb.heatmap(tGen, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.04*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.04*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.04*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%% Approx. Absorptionszeiten von Cooperation (wie oben)

# Umordnen
tC = np.flip(tC.reshape((N_i[0]+1,N_i[1]+1)),axis = 0)

sb.heatmap(tC, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.04*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.04*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.04*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

#%% Approx. Absorptionszeiten von Defection (wie oben)

# Umordnen
tD = np.flip(tD.reshape((N_i[0]+1,N_i[1]+1)),axis = 0)

sb.heatmap(tD, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.04*N_i[1],0.01*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 1.04*N_i[0], "D", va = 'center', ha = 'center')
plt.text(0.99*N_i[1],1.04*N_i[0],"C", va = 'center', ha = 'center')
plt.text(-0.04*N_i[1], 0.5*N_i[1], "Community 1", rotation = 90, ha = 'center', va = 'center')
plt.text(0.5*N_i[1], 1.04*N_i[0], "Community 2", va = 'center', ha = 'center')
plt.show()

end_time = time.time()
print()
print('Gesamtzeit: ', int((end_time - start_time)//60),'Minuten, ', int((end_time - start_time)%60), 'Sekunden.')
print('MC-Simulationen pro Zustand: ', tSpot, ' bei ', states, ' Zuständen')
print('Gesamtpopulation N: ', sum(N_i))