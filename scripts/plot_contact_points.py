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
    
    # Se c'è una sola colonna da plottare (come per l'osservatore di energia)
    if len(columns_to_plot) == 1:
        ax.plot(dataframe["Time"], dataframe[columns_to_plot[0]], label=legend_prefix)
    else: # Altrimenti, plotta tutte le componenti
        for col in columns_to_plot:
            # Estrai il numero della componente dal nome della colonna per la legenda
            component_index = col.strip().split('_')[-1]
            ax.plot(dataframe["Time"], dataframe[col], label=f'${legend_prefix}_{{{component_index}}}(t)$')
            
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize='18', framealpha=1.0)
    # Imposta i limiti dell'asse x come nell'immagine di esempio
    ax.set_xlim(0, 5)

# --- 2. CARICAMENTO DATI ---
# Assumiamo di avere file CSV separati per ogni set di dati
data = [None]*8
try:
    for i in range(0,8):
        data[i] = pd.read_csv(f"contact_point_{i}.csv")
except FileNotFoundError as e:
    print(f"Errore: file non trovato. Assicurati che tutti i file CSV siano presenti. Dettagli: {e}")
    exit()

# --- 3. CREAZIONE DELLA FIGURA E DEI SUBPLOT ---
# `subplots(5, 1)` crea 5 righe, 1 colonna di grafici.
# `figsize` controlla la dimensione totale della figura.
# `sharex=True` fa sì che tutti i subplot condividano lo stesso asse x (molto utile!).
fig, axes = plt.subplots(8, 1, figsize=(14, 20), sharex=True)

# --- 4. PLOT DI OGNI SUBPLOT ---
for i in range(0,8):
    plot_subplot(axes[i], data[i], f"Punto di contatto {i}", "", "p_", "p")\

# --- 5. RIFINITURE FINALI ---
# Aggiungi l'etichetta all'asse x solo all'ultimo subplot
axes[-1].set_xlabel("Tempo (s)")

# Ottimizza lo spazio tra i subplot
plt.tight_layout(pad=2.0)

# Salva la figura in alta qualità
plt.savefig("contact_points.png", dpi=300)

# Mostra il grafico
plt.show()