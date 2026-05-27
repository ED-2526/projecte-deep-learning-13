import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle

from dataloaders import get_dataloaders, BATCH_SIZE, AUGMENTATION_TYPE


# =========================
# CONFIG
# =========================

NUM_EPOCHS = 10
LR = 1e-3
PESOS_PATH = "pesos_bons_focal.pth"


# =========================
# FOCAL LOSS
# =========================

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        log_p = F.log_softmax(inputs, dim=1)

        ce = F.nll_loss(log_p, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-F.nll_loss(log_p, targets, reduction='none'))

        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# =========================
# HEAD FINAL (LA TEVA FC ORIGINAL)
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
# W&B
# =========================

nom_grafica = input("Nom de la gràfica a W&B: ")

wandb.init(
    project="ciudades-resnet18",
    name=nom_grafica,
    config={
        "epochs": NUM_EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "model": "efficientnet_b0",
        "loss": "FocalLoss",
        "augmentation": AUGMENTATION_TYPE
    }
)


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# DATA
# =========================

train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()
num_classes = len(class_names)

wandb.config.update({
    "num_classes": num_classes,
    "classes": class_names
})


# =========================
# MODEL
# =========================

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

in_features = model.classifier[1].in_features

# IMPORTANT: mantenim estructura estable
model.classifier[1] = FCFinal(in_features, num_classes)

model = model.to(device)


# =========================
# LOSS + OPTIMIZER
# =========================

criterion = FocalLoss(weight=class_weights.to(device))

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

wandb.watch(model, log="gradients", log_freq=50)


# =========================
# LOAD CHECKPOINT
# =========================

best_val_acc = 0.0
skip_training = False

if os.path.exists(PESOS_PATH):
    print(f"Carregant pesos des de {PESOS_PATH}")

    model.load_state_dict(
        torch.load(PESOS_PATH, map_location=device)
    )

    model.eval()
    skip_training = True

    print("Pesos carregats. Saltant entrenament.")


# =========================
# TRAIN
# =========================

if not skip_training:

    for epoch in range(NUM_EPOCHS):

        model.train()

        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = outputs.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)


        # =========================
        # VALIDATION
        # =========================

        model.eval()

        val_loss, val_correct, val_total = 0, 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = outputs.argmax(1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        wandb.log({
            "epoch": epoch + 1,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "acc/train": train_acc,
            "acc/val": val_acc
        })

        print(f"Epoch {epoch+1}")
        print(f"Train acc: {train_acc:.2f}% | Val acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), PESOS_PATH)
            wandb.save(PESOS_PATH)
            print("✔ Model guardat")


# =========================
# TEST
# =========================

model.eval()

all_outputs = []
all_labels = []

test_preds = []
test_labels = []

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:

        print("Processant batch de test...")

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(1)

        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())

        test_preds.extend(preds.cpu().tolist())
        test_labels.extend(labels.cpu().tolist())

        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

print("Test finalitzat. Processant resultats...")

all_outputs = torch.cat(all_outputs, dim=0)
all_labels = torch.cat(all_labels, dim=0)

test_acc = 100 * test_correct / test_total

print(f"Test accuracy: {test_acc:.2f}%")


# =========================
# PICKLE
# =========================

with open("test_results.pkl", "wb") as f:
    pickle.dump({
        "outputs": all_outputs,
        "labels": all_labels,
        "preds": test_preds
    }, f)

print("Results guardats a test_results.pkl")


# =========================
# CONFUSION MATRIX
# =========================

def log_cm(y_true, y_pred, class_names, title, key):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    wandb.log({key: wandb.Image(plt)})
    plt.close()


log_cm(
    test_labels,
    test_preds,
    class_names,
    "Test: Ciutats",
    "confusion_matrix/test"
)


# =========================
# FINAL LOGS
# =========================

wandb.log({
    "test_acc": test_acc,
    "best_val_acc": best_val_acc
})

wandb.run.summary["test_accuracy"] = test_acc
wandb.run.summary["best_val_accuracy"] = best_val_acc

torch.save(model.state_dict(), "last_model.pth")
wandb.save("last_model.pth")

wandb.finish()