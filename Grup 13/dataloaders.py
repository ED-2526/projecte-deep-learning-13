import os
import pickle
import torch
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

#CONFIGURACIÓ GENERAL

IMAGES_PATH = "/home/edxnG13/grup13/Images"
BATCH_SIZE = 128
IMG_SIZE = 224
SEED = 42
PKL_PATH = "/home/edxnG13/grup13/dataset.pkl"


def load_dataset(root_dir):

    """"
    Retorna una llista amb els camins de les imatges, les etiquetes associades i els noms de les classes.
    La llista de classes són els noms, però per pytorch necesstiem numeros no paraules i per això passem dues llistes, una amb els noms i una amb els numeros associats a cada classe.
    """
    image_paths = []
    labels = []

    class_names = sorted([
        folder for folder in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, folder))
    ])

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)

        for file in os.listdir(class_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_paths.append(os.path.join(class_dir, file))
                labels.append(label)

    return image_paths, labels, class_names


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths #LLista amb els paths de les imatges
        self.labels = labels #Llista amb etiquetes
        self.transform = transform #Transform és un objecte de torch que indica com es volen transformar les imatges

    def __len__(self):
        return len(self.image_paths) #Per calcular quantes imatges hi ha al dataset

    def __getitem__(self, index): #Funció que crida el dataloader per obtenir una imatge i la seva etiqueta associada
        image_path = self.image_paths[index] 
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB") #Convertim la imatge a RGB per assegurar-nos que totes les imatges tinguin 3 canals

        if self.transform:
            image = self.transform(image) #Li aplquem les transformacions que hem definit abans

        return image, label #Tornem imatge i etiqueta


def create_and_save_pickle():

    image_paths, labels, class_names = load_dataset(IMAGES_PATH) #obtenim les llistes de paths, labels i noms de classes

    #Definim mida train 70%, validació 15% i test 15%

    total_size = len(image_paths) 
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    #Aquí el problema era que no manteniem la proporció original de les classes entre els spits, per això ho fem d'una altra manera
    #indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(SEED)).tolist()
    #Fem servir la funció de sklearn per fer els splits, ja que ens permet mantenir la proporció de les classes entre els splits gràcies al paràmetre stratify, que li passem les etiquetes associades a cada imatge. Així ens assegurem que cada split tingui una representació equilibrada de les classes.
    
    
    indices = list(range(total_size)) #llista amb els índexs de les imatges, que van de 0 a total_size-1

    #train vs temp (val+test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.3,
        stratify=labels, #al passar-li les labels, aquestes tenen el mateix index que el de les imatges i així fem bé la separació
        random_state=SEED
    )

    # val vs test
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_size / (val_size + test_size),
        stratify=[labels[i] for i in temp_indices], #indiquem que només tingui en compte els labels associats als indexs que estem fent servir
        random_state=SEED
    )

    #Guardem en un diccionari i ho guardem en un fitxer pickle
    data = {
        "image_paths": image_paths,
        "labels": labels,
        "class_names": class_names,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices
    }

    with open(PKL_PATH, "wb") as f:
        pickle.dump(data, f)

    print("Pickle guardado en:", PKL_PATH)

#Carreguem pickle
def load_pickle():
    if not os.path.exists(PKL_PATH):
        create_and_save_pickle()

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    return data

#Per no haver de tornar a llegir la carpeta d'imatges tota l'estona i com que tots tindrem la mateixa carpeta superior
#és a dir, tots tindrem les mateixes rutes a les fotos així millor
def get_dataloaders():

    #Tornem a carregar les dades del pickle
    data = load_pickle()

    image_paths = data["image_paths"]
    labels = data["labels"]
    class_names = data["class_names"]

    train_indices = data["train_indices"]
    val_indices = data["val_indices"]
    test_indices = data["test_indices"]

    #Definim quines transformacions farem
    transform = transforms.Compose([ #amb el compose indiquem que volem aplicar les transformacions una rere l'altra
        transforms.Resize((IMG_SIZE, IMG_SIZE)), #fem un resize
        transforms.ToTensor() #convertim a tensor, el que fem és passem els valor de 0-255 a 0-1 i canviem l'ordre de les dimensions de (H, W, C) a (Canal, H, W) que és el que espera pytorch
    ])

    #Creem l'objecte dataset
    dataset = ImageDataset(
        image_paths=image_paths,
        labels=labels,
        transform=transform
    )

    #Indiquem amb quins indexs dels dataset volem fer el train, val i test
    #El que fem és indicar que quan em demanin dades de train per exemple només agafo aquests index que he indicat 
    train_dataset = Subset(dataset, train_indices) 
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    #GESTIONEM QUE ELS MINIBATCHS TINGUIN UNA PROPORCIÓ EQUILIBRADA DE LES CLASSES

    #Comptem quants exemples hi ha per cada classe al train per calcular els pesos de cada classe 
    train_labels = [labels[i] for i in train_indices]
    label_counts = Counter(train_labels)

    #Calculem un tensor amb els pesos associats a cada classe que és el que li passarem a la funció de loss 

    #li assignem un pes a cada classe que és inversament proporcional al nombre d'exemples que té al train
    class_weights = { 
        class_id: 1.0 / count
        for class_id, count in label_counts.items()
    }

    
    weights_tensor = torch.tensor(
    [class_weights[i] for i in range(len(class_weights))],
    dtype=torch.float
    ) #convertim a tensor perquè és el format que espera pytorch, mida és el número de classes i cada valor és el pes associat a cada classe

    #Fem els dataloaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True  
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, class_names, weights_tensor
