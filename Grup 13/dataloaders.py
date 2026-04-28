import os
import pickle
import torch
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms

#prova abel

IMAGES_PATH = "Images"
BATCH_SIZE = 32
IMG_SIZE = 224
SEED = 42
PKL_PATH = "dataset_splits.pkl"


def load_dataset(root_dir):
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
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def create_and_save_pickle():
    image_paths, labels, class_names = load_dataset(IMAGES_PATH)

    total_size = len(image_paths)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(SEED)).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

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


def load_pickle():
    if not os.path.exists(PKL_PATH):
        create_and_save_pickle()

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    return data


def get_dataloaders():
    data = load_pickle()

    image_paths = data["image_paths"]
    labels = data["labels"]
    class_names = data["class_names"]

    train_indices = data["train_indices"]
    val_indices = data["val_indices"]
    test_indices = data["test_indices"]

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    dataset = ImageDataset(
        image_paths=image_paths,
        labels=labels,
        transform=transform
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_labels = [labels[i] for i in train_indices]
    label_counts = Counter(train_labels)

    class_weights = {
        class_id: 1.0 / count
        for class_id, count in label_counts.items()
    }

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler
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

    return train_loader, val_loader, test_loader, class_names
