import os
import pickle
import torch
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# CONFIGURACIÓ GENERAL
IMAGES_PATH = "/home/edxnG13/grup13/Images"
BATCH_SIZE = 32
IMG_SIZE = 224
SEED = 42
PKL_PATH = "/home/edxnG13/grup13/dataset.pkl"

# num_workers indica quants processos de CPU carreguen imatges en paral·lel.
# 4 és un valor bastant habitual: accelera la càrrega sense saturar massa la màquina.
NUM_WORKERS = 4

# pin_memory ajuda a copiar dades més ràpid de CPU a GPU.
# Només té sentit activar-ho si hi ha CUDA disponible.
PIN_MEMORY = torch.cuda.is_available()


def load_dataset(root_dir):
    """
    Retorna una llista de mostres, els noms de les classes i el diccionari classe -> índex.

    Cada mostra és una tupla:
        (camí_de_la_imatge, etiqueta_numèrica)

    Així evitem tenir dues llistes separades, image_paths i labels.
    """

    samples = []

    class_names = sorted([
        folder for folder in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, folder))
    ])

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(class_names)
    }

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        label = class_to_idx[class_name]

        for file in os.listdir(class_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_path = os.path.join(class_dir, file)
                samples.append((image_path, label))

    return samples, class_names, class_to_idx


class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples      # Llista de tuples: (path, label)
        self.transform = transform  # Transformacions de torchvision

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")  # assegurem 3 canals

        if self.transform:
            image = self.transform(image)

        return image, label


def create_and_save_pickle():
    samples, class_names, class_to_idx = load_dataset(IMAGES_PATH)

    total_size = len(samples)
    indices = list(range(total_size))

    # Extraiem només les etiquetes des de samples per poder fer el split estratificat.
    labels = [sample[1] for sample in samples]

    # Split estratificat: manté la proporció de classes entre train i temp.
    # 70% train i 30% temporal, que després dividirem en val i test.
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.3,
        stratify=labels,
        random_state=SEED
    )

    # Dividim el 30% temporal en 15% validació i 15% test.
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        stratify=[labels[i] for i in temp_indices],
        random_state=SEED
    )

    data = {
        "samples": samples,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices
    }

    with open(PKL_PATH, "wb") as f:
        pickle.dump(data, f)

    print("Pickle guardat en:", PKL_PATH)


def load_pickle():
    if not os.path.exists(PKL_PATH):
        create_and_save_pickle()

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    return data


def get_dataloaders():
    data = load_pickle()

    samples = data["samples"]
    class_names = data["class_names"]

    train_indices = data["train_indices"]
    val_indices = data["val_indices"]
    test_indices = data["test_indices"]

    # Transformacions aplicades a totes les imatges.
    # Resize: adapta totes les imatges a 224x224, mida habitual per ResNet.
    # ToTensor: passa els píxels de 0-255 a 0-1 i canvia l'ordre a (C, H, W).
    # Normalize: aplica (pixel - mean) / std per canal RGB.
    #
    # Els valors mean i std són els d'ImageNet:
    # mean=[0.485, 0.456, 0.406] → mitjana dels canals R, G i B.
    # std=[0.229, 0.224, 0.225] → desviació estàndard dels canals R, G i B.
    #
    # Ho fem perquè ResNet18 està preentrenada amb ImageNet i espera entrades
    # amb una distribució semblant a la que va veure durant el seu entrenament original.
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageDataset(
        samples=samples,
        transform=transform
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Comptem quants exemples hi ha de cada classe al train.
    train_labels = [samples[i][1] for i in train_indices]
    label_counts = Counter(train_labels)

    # Pes invers proporcional al nombre d'exemples:
    # si una classe té menys imatges, tindrà més pes a la loss.
    class_weights = {
        class_id: 1.0 / count
        for class_id, count in label_counts.items()
    }

    # Convertim els pesos a tensor perquè CrossEntropyLoss els pugui utilitzar.
    weights_tensor = torch.tensor(
        [class_weights[i] for i in range(len(class_names))],
        dtype=torch.float
    )

    # DataLoader de train:
    # shuffle=True barreja les mostres a cada epoch.
    # num_workers carrega imatges en paral·lel.
    # pin_memory accelera el pas CPU -> GPU.
    # persistent_workers manté els workers vius entre epochs i evita recrear-los.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, class_names, weights_tensor