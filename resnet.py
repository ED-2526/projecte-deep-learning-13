import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import wandb

from dataloaders import get_dataloaders

# =========================
# CONFIG
# =========================

NUM_EPOCHS = 10
LR = 1e-4

wandb.init(
    project="ciudades-resnet18",
    name="resnet18-transfer-learning",
    config={
        "epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "batch_size": 32,
        "model": "resnet18",
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss"
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# DATALOADERS
# =========================

train_loader, val_loader, test_loader, class_names = get_dataloaders()
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
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# =========================
# MODEL
# =========================

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.fc = FCFinal(model.fc.in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

wandb.watch(model, criterion, log="all", log_freq=10)

# =========================
# TRAIN
# =========================

for epoch in range(NUM_EPOCHS):
    model.train()

    train_loss_total = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        print("Nou batch d'entrenament:", images.size(0))

        images = images.to(device)
        labels = labels.to(device)

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

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item()

            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_loss = val_loss_total / len(val_loader)
    val_acc = 100 * val_correct / val_total

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print("-" * 40)

# =========================
# TEST FINAL
# =========================

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        test_total += labels.size(0)
        test_correct += (preds == labels).sum().item()

test_acc = 100 * test_correct / test_total

wandb.log({
    "test_accuracy": test_acc
})

print(f"Test Accuracy: {test_acc:.2f}%")

# =========================
# GUARDAR MODELO
# =========================

torch.save(model.state_dict(), "resnet18_ciudades.pth")

wandb.save("resnet18_ciudades.pth")

wandb.finish()