# Evolutionäre Spiele auf Community-basierten Graphen
Dieses Repository enthält die Implementierungen, die zu meiner Masterarbeit zum Thema "Evolutionäre Spiele auf Community-basierten Graphen" selbst erstellt wurden.

# Modell
Es werden zwei verschiedene Modelle betrachtet:
- Ein angepasstes Populationsspiel für sub-homogenen Populationen (SHPG)
- Das Stochastische Blockmodell (SBM)
Für diese Modelle werden evolutionäre Spiele, im Wesentlichen Kooperationsspiele, betrachtet. Dabei werden folgende Update-Regeln genutzt:
- Geglättete Imitation mit Sigmoid-Funktion (SIUR, "Smoothened Imitation Update-Rule")
- Geglättete Imitation mit Mutation (MSIUR)
- Logit-Regel mit Boltzman-Funktion (LogUR)

# Spiele
Es werden primär Kooperationsspiele (CG) betrachtet.
Außerdem sind MSIUR und LogUR für das Spiel Schere-Stein-Papier (RPS) implementiert.

# Monte-Carlo-Simulationen
Die Simulationsprogramme sind für zwei Communities ausgelegt.
- SIUR: Die Monte-Carlo-Simulation approximiert die Absorptionswahrscheinlichkeiten ausgehend von den Startzuständen
- MSIUR, LogUR: Die Monte-Carlo-Simulationen approximieren die stationäre Verteilung

# Analytische Methoden
- FMM: Für die SIUR wird die Fundamentalmatrixmethode genutzt, um die Absorptionswahrscheinlichkeiten und -zeiten zu bestimmen. Ausgelegt für zwei Communities (Laufzeit).
- StatDist: Für die MSIUR und LogUR wird über dünnbesetzte Matrizen die Eigenvektorberechnung ausgeführt, um die stationäre Verteilung zu berechnen. Für Kooperationsspiele sind mehrere Communities möglich, visuelle Darstellung ist für bis zu drei Communities implementiert. Für RPS werden aufgrund der Laufzeit maximal zwei Communities empfohlen.
- MT: Für alle drei Update-Regeln werden für Koordinationsspiele die momentanen Trajektorien dargestellt. Ausgelegt für zwei Communities (visuelle Darstellung).
