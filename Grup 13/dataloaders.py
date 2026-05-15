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
PKL_PATH = "/home/edxnG13/grup13/dataset_pickle.pkl"

# Canvia aquest valor per provar diferents data augmentations:
# "none"
# "flip"
# "jitter"
# "crop"
# "rotation"
# "perspective"
# "blur"
# "affine"
# "flip_jitter"
# "crop_flip_jitter"
# "affine_flip_jitter"
# "crop_flip_jitter_rotation"
AUGMENTATION_TYPE = "rotation"

# num_workers indica quants processos de CPU carreguen imatges en paral·lel.
# 4 és un valor bastant habitual: accelera la càrrega sense saturar massa la màquina.
NUM_WORKERS = 4

# pin_memory ajuda a copiar dades més ràpid de CPU a GPU.
# Només té sentit activar-ho si hi ha CUDA disponible.
PIN_MEMORY = torch.cuda.is_available()


def load_dataset(root_dir):
    """
    Retorna una llista de mostres, els noms de les classes i el diccionari classe -> índex.

    Cada mostra és una tupla (samples):
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


def get_train_transform():
    """
    Retorna les transformacions del train segons AUGMENTATION_TYPE.
    Només el train té data augmentation, perquè és on el model aprèn.
    Validation i test han de ser estables i no tenir transformacions aleatòries.
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if AUGMENTATION_TYPE == "none":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "flip":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "jitter":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3            ),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "crop":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                IMG_SIZE,
                scale=(0.8, 1.0)
            ),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "rotation":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "perspective":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.5
            ),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "blur":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.GaussianBlur(
                kernel_size=3
            ),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "affine":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "flip_jitter":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3
            ),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "crop_flip_jitter":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                IMG_SIZE,
                scale=(0.8, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "affine_flip_jitter":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3
            ),
            transforms.ToTensor(),
            normalize
        ])

    elif AUGMENTATION_TYPE == "crop_flip_jitter_rotation":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                IMG_SIZE,
                scale=(0.8, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3
            ),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize
        ])

    else:
        raise ValueError(f"Augmentation no reconeguda: {AUGMENTATION_TYPE}")


def get_eval_transform():
    """
    Transformacions de validation i test.
    No tenen data augmentation perquè volem avaluar sempre igual.
    """

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_dataloaders():
    data = load_pickle()

    samples = data["samples"]
    class_names = data["class_names"]

    train_indices = data["train_indices"]
    val_indices = data["val_indices"]
    test_indices = data["test_indices"]

    train_transform = get_train_transform()
    eval_transform = get_eval_transform()

    # Creem dos datasets base:
    # un amb augmentation per train i un sense augmentation per validation/test.
    train_dataset_full = ImageDataset(
        samples=samples,
        transform=train_transform
    )

    eval_dataset_full = ImageDataset(
        samples=samples,
        transform=eval_transform
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(eval_dataset_full, val_indices)
    test_dataset = Subset(eval_dataset_full, test_indices)

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


def show_augmentation_example(index=0):
    """
    Mostra una imatge original i la mateixa imatge després d'aplicar-li
    l'augmentation actual. Serveix per visualitzar què està fent cada prova.
    """

    data = load_pickle()
    samples = data["samples"]

    image_path, label = samples[index]

    original_image = Image.open(image_path).convert("RGB")
    augmented_image = get_train_transform()(original_image)

    # Desnormalitzem per poder visualitzar correctament la imatge transformada.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    augmented_image = augmented_image * std + mean
    augmented_image = torch.clamp(augmented_image, 0, 1)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image.permute(1, 2, 0))
    plt.title(f"Augmentation: {AUGMENTATION_TYPE}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("augmentation_preview.png")
    print("Imatge guardada com augmentation_preview.png")