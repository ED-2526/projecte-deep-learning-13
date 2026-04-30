# 1. Definició (assegura't que això estigui a dalt)
mapping_continents = {
    'Bangkok': 'Àsia', 'Osaka': 'Àsia',
    'Barcelona': 'Europa', 'Brussels': 'Europa', 'Lisbon': 'Europa', 
    'London': 'Europa', 'Madrid': 'Europa', 'OSL': 'Europa', 
    'PRG': 'Europa', 'PRS': 'Europa', 'Rome': 'Europa',
    'Boston': 'Amèrica del Nord', 'Chicago': 'Amèrica del Nord', 
    'LosAngeles': 'Amèrica del Nord', 'MexicoCity': 'Amèrica del Nord', 
    'Miami': 'Amèrica del Nord', 'Minneapolis': 'Amèrica del Nord', 
    'Phoenix': 'Amèrica del Nord', 'TRT': 'Amèrica del Nord', 
    'WashingtonDC': 'Amèrica del Nord',
    'BuenosAires': 'Amèrica del Sud', 'Medellin': 'Amèrica del Sud',
    'Melbourne': 'Oceania'
}


# 2. NOVA FUNCIÓ: Retorna l'ÍNDEX (0, 1, 2...) en comptes del text
def to_continent_index_list(indices, classes, mapping, cont_names):
    # Agafa l'índex de la ciutat -> Nom ciutat -> Nom continent -> Índex del continent
    return [cont_names.index(mapping[classes[i]]) for i in indices]
