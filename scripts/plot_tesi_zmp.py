import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. FUNZIONE DI PLOT AUSILIARIA (per non ripetere il codice) ---
def plot_subplot(ax, dataframe, title, ylabel, y_cols_prefix, legend_prefix):
    """
    Funzione per plottare un singolo subplot.
    - ax: L'asse matplotlib su cui disegnare.
    - dataframe: Il DataFrame pandas con i dati.
    - title: Il titolo del subplot.
    - ylabel: L'etichetta dell'asse y.
    - y_cols_prefix: Il prefisso delle colonne da plottare (es. 'tau_ext_').
    """
    # Trova tutte le colonne che iniziano con il prefisso
    columns_to_plot = [col for col in dataframe.columns if col.strip().startswith(y_cols_prefix)]
    
    # Se c'è una sola colonna da plottare (come per l'osservatore di energia o collision_link)
    if len(columns_to_plot) == 1:
        ax.plot(dataframe["Time"], dataframe[columns_to_plot[0]], label=legend_prefix)
    else: # Altrimenti, plotta tutte le componenti
        for col in columns_to_plot:
            # Estrai il numero del giunto dal nome della colonna per la legenda
            # Assumiamo che il formato sia 'prefisso_numero'
            try:
                joint_index = col.strip().split('_')[-1]
                # Controllo base per vedere se è un numero, altrimenti usa il nome colonna
                int(joint_index) 
                label = f'${legend_prefix}_{{{joint_index}}}(t)$'
            except ValueError:
                label = col # Fallback se non riesce a parsare l'indice
            
            ax.plot(dataframe["Time"], dataframe[col], label=label)
            
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize='18', framealpha=1.0)
    # Imposta i limiti dell'asse x come nell'immagine di esempio
    ax.set_xlim(9, 22)
    ax.set_ylim(-0.2, 0.7)

# --- 2. CARICAMENTO DATI ---
# Assumiamo di avere file CSV separati per ogni set di dati
# Sostituisci i nomi dei file con i tuoi se diversi
try:
    zmp = pd.read_csv("ZMP.csv")
    com = pd.read_csv("COM.csv")

except FileNotFoundError as e:
    print(f"Errore: file non trovato. Assicurati che tutti i file CSV siano presenti. Dettagli: {e}")


# --- 3. CREAZIONE DELLA FIGURA E DEI SUBPLOT ---
# `subplots(3, 2)` crea 3 righe e 2 colonne di grafici.
# `figsize` modificato per adattarsi al layout a griglia (più largo, meno alto).
# `sharex=True` condivide l'asse X tra le colonne.
fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# `axes` è ora una matrice 3x2 di oggetti Axes.
# Possiamo appiattirla in un array 1D per iterarci sopra facilmente: axes.flat
# Oppure accedervi come axes[riga, colonna].

# --- 4. PLOT DI OGNI SUBPLOT ---

# Grafico 1 (Riga 0, Colonna 0)
plot_subplot(axes[0], zmp, r"Ricostruzione dello ZMP sul robot reale", "m", "p_", "p")

# Grafico 2 (Riga 0, Colonna 1)
plot_subplot(axes[1], com, r" Posizione del COM del robot reale", "m", "p_", "p")


# --- 5. RIFINITURE FINALI ---
# Aggiungi l'etichetta all'asse x solo per l'ultima riga (condivisa)
axes[1].set_xlabel("Tempo (s)", fontsize=16)

# Ottimizza lo spazio tra i subplot per evitare sovrapposizioni
plt.tight_layout(pad=2.0)

# Salva la figura
plt.savefig("grafici_zmp.png", dpi=300)

# Mostra il grafico
plt.show()