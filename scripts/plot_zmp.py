import pandas as pd
import matplotlib.pyplot as plt

# Carica i dati dal file CSV. pandas usa la prima riga come intestazione di default.
try:
    df = pd.read_csv("ZMP.csv")
except FileNotFoundError:
    print("Errore: file 'zmp.csv' non trovato. Esegui prima il programma C++.")
    exit()

# Stampa le prime 5 righe per controllare che sia tutto a posto
print("Dati caricati:")
print(df.head())

# Crea una figura e un set di assi
plt.figure(figsize=(12, 8))
# Plotta ogni colonna (componente del vettore 'q') contro il tempo
# pandas permette di plottare usando direttamente i nomi delle colonne
for i in range(1, 4):
    plt.plot(df["Time"], df[f" p_{i}"], label=f"Componente p{i}")


# Aggiungi etichette, titolo e legenda per rendere il grafico leggibile

plt.title("Mujoco Wrench applicato al piede sinistro")
plt.xlabel("Tempo (s)")
plt.ylabel("Valore Componente")
plt.legend()  # Mostra la legenda con le etichette
plt.grid(True) # Aggiunge una griglia per una migliore leggibilità
plt.xlim(9, 22)
# Mostra il grafico
plt.show()
