import os
import pickle
import torch
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================

IMAGES_PATH = "/home/edxnG13/grup13/Images"
BATCH_SIZE = 32
IMG_SIZE = 224
SEED = 42
PKL_PATH = "/home/edxnG13/grup13/dataset_pickle.pkl"

AUGMENTATION_TYPE = "crop_flip_jitter"

NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()


# =========================
# DATASET LOADER
# =========================

def load_dataset(root_dir):

    samples = []

    class_names = sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
    ])

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        label = class_to_idx[class_name]

        for file in os.listdir(class_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                samples.append((os.path.join(class_dir, file), label))

    return samples, class_names, class_to_idx


class ImageDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# PICKLE
# =========================

def create_and_save_pickle():

    samples, class_names, class_to_idx = load_dataset(IMAGES_PATH)

    indices = list(range(len(samples)))
    labels = [s[1] for s in samples]

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=labels,
        random_state=SEED
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=[labels[i] for i in temp_idx],
        random_state=SEED
    )

    data = {
        "samples": samples,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx
    }

    with open(PKL_PATH, "wb") as f:
        pickle.dump(data, f)


def load_pickle():

    if not os.path.exists(PKL_PATH):
        create_and_save_pickle()

    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


# =========================
# TRANSFORMS FAST (IMPORTANT)
# =========================

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def get_train_transform():

    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        normalize
    ])


def get_eval_transform():

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])


# =========================
# DATALOADERS FAST
# =========================

def get_dataloaders():

    data = load_pickle()

    samples = data["samples"]
    class_names = data["class_names"]

    train_idx = data["train_indices"]
    val_idx = data["val_indices"]
    test_idx = data["test_indices"]

    train_ds = Subset(
        ImageDataset(samples, get_train_transform()),
        train_idx
    )

    eval_ds = Subset(
        ImageDataset(samples, get_eval_transform()),
        val_idx
    )

    test_ds = Subset(
        ImageDataset(samples, get_eval_transform()),
        test_idx
    )

    train_labels = [samples[i][1] for i in train_idx]
    counts = Counter(train_labels)

    weights = torch.tensor(
        [1.0 / counts[i] for i in range(len(class_names))],
        dtype=torch.float
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": True,
        "prefetch_factor": 2
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        eval_ds,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_kwargs
    )

    return train_loader, val_loader, test_loader, class_names, weights