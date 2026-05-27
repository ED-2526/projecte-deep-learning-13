import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models import vit_b_16, ViT_B_16_Weights

import wandb

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from dataloaders_vit import (
    get_dataloaders,
    BATCH_SIZE,
    AUGMENTATION_TYPE
)

# =========================
# SPEED OPTIMIZATIONS
# =========================

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================
# CONFIG
# =========================

NUM_EPOCHS = 10
LR = 3e-5

BEST_MODEL_PATH = "vit_best.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

nom_grafica = input("Nom de la gràfica a W&B: ")

# =========================
# WANDB
# =========================

wandb.init(
    project="ciudades-resnet18",
    name=nom_grafica,
    config={
        "epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "model": "vit_b_16",
        "optimizer": "AdamW",
        "loss": "FocalLoss",
        "augmentation": AUGMENTATION_TYPE,
        "amp": True
    }
)

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

        ce = F.nll_loss(
            log_p,
            targets,
            reduction='none',
            weight=self.weight
        )

        pt = torch.exp(
            -F.nll_loss(
                log_p,
                targets,
                reduction='none'
            )
        )

        loss = ((1 - pt) ** self.gamma) * ce

        return loss.mean()

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
# HEAD FINAL
# =========================

class FCFinal(nn.Module):

    def __init__(self, in_features, num_classes):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(in_features, 96),

            nn.BatchNorm1d(96),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(96, num_classes)
        )

    def forward(self, x):

        return self.classifier(x)

# =========================
# MODEL
# =========================

model = vit_b_16(
    weights=ViT_B_16_Weights.DEFAULT
)

in_features = model.heads.head.in_features

model.heads.head = FCFinal(
    in_features,
    num_classes
)

model = model.to(device)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = FocalLoss(
    weight=class_weights.to(device)
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# =========================
# AMP
# =========================

scaler = torch.amp.GradScaler('cuda')

# =========================
# WANDB WATCH
# =========================

wandb.watch(
    model,
    criterion,
    log="all",
    log_freq=50
)

# =========================
# SKIP TRAINING SI JA HI HA PESOS
# =========================

skip_training = False

best_val_acc = 0.0

if os.path.exists(BEST_MODEL_PATH):

    print(f"\n🔥 Carregant pesos des de {BEST_MODEL_PATH}")

    model.load_state_dict(
        torch.load(
            BEST_MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    skip_training = True

    print("✅ Pesos carregats correctament")
    print("⏩ Saltant entrenament")

# =========================
# TRAIN LOOP
# =========================

if not skip_training:

    for epoch in range(NUM_EPOCHS):

        # =========================
        # TRAIN
        # =========================

        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for i, (images, labels) in enumerate(train_loader):

            if i % 100 == 0:

                print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Batch {i}/{len(train_loader)}"
                )

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda'):

                outputs = model(images)

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

            preds = outputs.argmax(1)

            train_correct += (
                preds == labels
            ).sum().item()

            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total

        train_loss /= len(train_loader)

        # =========================
        # VALIDATION
        # =========================

        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast(device_type='cuda'):

                    outputs = model(images)

                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = outputs.argmax(1)

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        val_loss /= len(val_loader)

        # =========================
        # WANDB LOG
        # =========================

        wandb.log({

            "epoch": epoch + 1,

            "loss/train": train_loss,

            "loss/validation": val_loss,

            "accuracy/train": train_acc,

            "accuracy/validation": val_acc
        })

        print(
            f"\nEpoch {epoch+1}/{NUM_EPOCHS}"
        )

        print(
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # =========================
        # SAVE BEST MODEL
        # =========================

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            wandb.save(BEST_MODEL_PATH)

            print(
                f"🔥 Nou millor model: "
                f"{best_val_acc:.2f}%"
            )

# =========================
# LOAD BEST MODEL
# =========================

print("\n📦 Carregant millor model...")

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)

model.eval()

# =========================
# TEST
# =========================

print("\n===== TEST FINAL =====")

all_preds = []
all_labels = []

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(device_type='cuda'):

            outputs = model(images)

        preds = outputs.argmax(1)

        all_preds.extend(
            preds.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        test_correct += (
            preds == labels
        ).sum().item()

        test_total += labels.size(0)

test_acc = 100 * test_correct / test_total

print(f"\n🚀 TEST ACCURACY: {test_acc:.2f}%")

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

wandb.log({
    "confusion_matrix": wandb.Image(
        "confusion_matrix.png"
    )
})

plt.close()

# =========================
# FINAL LOGS
# =========================

wandb.log({
    "accuracy/test": test_acc
})

wandb.run.summary["test_accuracy"] = test_acc

wandb.run.summary["augmentation"] = AUGMENTATION_TYPE

# =========================
# SAVE FINAL MODEL
# =========================

torch.save(
    model.state_dict(),
    "vit_last.pth"
)

wandb.save("vit_last.pth")

# =========================
# FINISH
# =========================

wandb.finish()