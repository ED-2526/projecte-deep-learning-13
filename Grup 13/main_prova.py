import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import wandb
from dataloaders import get_dataloaders, BATCH_SIZE, AUGMENTATION_TYPE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# =========================
# CONFIG
# =========================

NUM_EPOCHS = 10

# LR = 1e-4
LR = 5e-4
# LR = 1e-3

PESOS_PATH = "pesos_bons.pth"

nom_grafica = input("Nom de la gràfica a W&B: ")

wandb.init(
    project="ciudades-resnet18",
    name=nom_grafica,
    config={
        "epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "model": "efficientnet_b0",
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
        "fc": "Linear(96)-BatchNorm-ReLU-Dropout(0.3)-Linear",
        "augmentation": AUGMENTATION_TYPE
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# DATALOADERS
# =========================

train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()

num_classes = len(class_names)

wandb.config.update({
    "num_classes": num_classes,
    "classes": class_names
})

# =========================
# HEAD FINAL
# =========================

class FCFinal(nn.Module):

    def __init__(self, in_features, num_classes):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(96, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# =========================
# MODEL
# =========================

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

in_features = model.classifier[1].in_features

model.classifier[1] = FCFinal(in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

wandb.watch(model, criterion, log="all", log_freq=10)

best_val_acc = 0.0
best_val_preds = []
best_val_labels = []

# =========================
# CARREGAR PESOS SI EXISTEIXEN
# =========================

skip_training = False

if os.path.exists(PESOS_PATH):

    print(f"Carregant pesos des de {PESOS_PATH}")

    model.load_state_dict(
        torch.load(PESOS_PATH, map_location=device)
    )

    model.eval()

    skip_training = True

    print("Pesos carregats correctament")
    print("Saltant entrenament...")

# =========================
# TRAIN
# =========================

if not skip_training:

    for epoch in range(NUM_EPOCHS):

        model.train()

        train_loss_total = 0
        train_correct = 0
        train_total = 0

        for i, (images, labels) in enumerate(train_loader):

            if (i + 1) % 200 == 0:
                print(f"Batch {i+1} | mida batch: {images.size(0)}")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss_total += loss.item()

            _, preds = torch.max(outputs, 1)

            train_total += labels.size(0)

            train_correct += (preds == labels).sum().item()

        train_loss = train_loss_total / len(train_loader)

        train_acc = 100 * train_correct / train_total

        # =========================
        # VALIDATION
        # =========================

        model.eval()

        val_loss_total = 0
        val_correct = 0
        val_total = 0

        current_val_preds = []
        current_val_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss_total += loss.item()

                _, preds = torch.max(outputs, 1)

                val_total += labels.size(0)

                val_correct += (preds == labels).sum().item()

                current_val_preds.extend(preds.cpu().tolist())
                current_val_labels.extend(labels.cpu().tolist())

        val_loss = val_loss_total / len(val_loader)

        val_acc = 100 * val_correct / val_total

        wandb.log({
            "epoch": epoch + 1,
            "loss/train": train_loss,
            "loss/validation": val_loss,
            "accuracy/train": train_acc,
            "accuracy/validation": val_acc,
        })

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        print("-" * 40)

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_val_preds = current_val_preds
            best_val_labels = current_val_labels

            torch.save(model.state_dict(), PESOS_PATH)

            wandb.save(PESOS_PATH)

            print(f"Nou millor model guardat amb Val Acc: {best_val_acc:.2f}%")

# =========================
# TEST FINAL
# =========================

model.eval()

test_correct = 0
test_total = 0

all_test_preds = []
all_test_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        test_total += labels.size(0)

        test_correct += (preds == labels).sum().item()

        all_test_preds.extend(preds.cpu().tolist())
        all_test_labels.extend(labels.cpu().tolist())

test_acc = 100 * test_correct / test_total

# =========================
# CONFUSION MATRIX
# =========================

def log_cm(y_true, y_pred, class_names, title, key, normalize=False):

    cm = confusion_matrix(
        y_true,
        y_pred,
        normalize="true" if normalize else None
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(title)

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    wandb.log({key: wandb.Image(plt)})

    plt.close()

# =========================
# LOGS
# =========================

wandb.log({
    "accuracy/test": test_acc,
    "accuracy/best_validation": best_val_acc,
})

log_cm(
    all_test_labels,
    all_test_preds,
    class_names,
    "Test: Ciutats",
    "confusion_matrix/cities/test"
)

print(f"Test Accuracy: {test_acc:.2f}%")

if not skip_training:
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")

wandb.run.summary["test_accuracy"] = test_acc
wandb.run.summary["augmentation"] = AUGMENTATION_TYPE

if not skip_training:
    wandb.run.summary["best_val_accuracy"] = best_val_acc

torch.save(model.state_dict(), "last_efficientnet_b0_ciudades.pth")

wandb.save("last_efficientnet_b0_ciudades.pth")

wandb.finish()