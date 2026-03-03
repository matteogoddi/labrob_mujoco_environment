import pandas as pd
import matplotlib.pyplot as plt

# Carica i dati dal file CSV. pandas usa la prima riga come intestazione di default.
try:
    df = pd.read_csv("errore_residuo_simulazione.csv")
except FileNotFoundError:
    print("Errore: file 'error_residuo_simulazione.csv' non trovato. Esegui prima il programma C++.")
    exit()

# Stampa le prime 5 righe per controllare che sia tutto a posto
print("Dati caricati:")
print(df.head())

# --- Creazione del Plot ---
for j in range(1,7):
    # Crea una figura e un set di assi
    plt.figure(figsize=(12, 8))

    # Plotta ogni colonna (componente del vettore 'r') contro il tempo
    # pandas permette di plottare usando direttamente i nomi delle colonne
    for i in range(j*5-4, j*5+1):
        plt.plot(df["Time"], df[f" r_{i}"], label=f"Componente r{i}")

    # Aggiungi etichette, titolo e legenda per rendere il grafico leggibile
    plt.title("Evoluzione delle Componenti del Vettore 'r' nel Tempo")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Valore Componente")
    plt.legend()  # Mostra la legenda con le etichette
    plt.grid(True) # Aggiunge una griglia per una migliore leggibilità

    # Mostra il grafico
    plt.show()

plt.figure(figsize=(12, 8))
# Plotta ogni colonna (componente del vettore 'r') contro il tempo
# pandas permette di plottare usando direttamente i nomi delle colonne
for i in range(31, 34):
    plt.plot(df["Time"], df[f" r_{i}"], label=f"Componente r{i}")
# Aggiungi etichette, titolo e legenda per rendere il grafico leggibile
plt.title("Evoluzione delle Componenti del Vettore 'r' nel Tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Valore Componente")
plt.legend()  # Mostra la legenda con le etichette
plt.grid(True) # Aggiunge una griglia per una migliore leggibilità
# Mostra il grafico
plt.show()
