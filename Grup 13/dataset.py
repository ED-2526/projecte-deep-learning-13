import os
import matplotlib.pyplot as plt

def graficar_imatges_per_ciutat(ruta_base):
    ciutats = []
    recompte_imatges = []

    # Comprovem si la carpeta Images existeix
    if not os.path.exists(ruta_base):
        print(f"Error: La carpeta '{ruta_base}' no existeix.")
        return

    # Iterem per cada subcarpeta (ciutat) dins de Images
    # Ordenem alfabèticament per a una millor visualització
    for ciutat in sorted(os.listdir(ruta_base)):
        ruta_ciutat = os.path.join(ruta_base, ciutat)
        
        # Ens assegurem que sigui un directori
        if os.path.isdir(ruta_ciutat):
            # Comptem quants fitxers hi ha (pots filtrar per extensions si cal)
            fitxers = [f for f in os.listdir(ruta_ciutat) if os.path.isfile(os.path.join(ruta_ciutat, f))]
            
            ciutats.append(ciutat)
            recompte_imatges.append(len(fitxers))

    # Creació del gràfic
    plt.figure(figsize=(12, 6))
    bars = plt.bar(ciutats, recompte_imatges, color='skyblue', edgecolor='navy')
    
    # Afegim els números sobre les barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', va='bottom')

    # Personalització del gràfic
    plt.xlabel('Ciutats')
    plt.ylabel('Nombre d\'imatges')
    plt.title('Distribució d\'imatges per ciutat')
    plt.xticks(rotation=45, ha='right')  # Rotem els noms per a que es llegueixin millor
    plt.tight_layout()

    # Mostrem el gràfic
    plt.show()

# Executem la funció
ruta_images = "Images"  # Ajusta aquesta ruta si el script no està a l'arrel
graficar_imatges_per_ciutat(ruta_images)
