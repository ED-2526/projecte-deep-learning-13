import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import wandb
from dataloaders import get_dataloaders, BATCH_SIZE

# =========================
# CONFIG
# =========================

NUM_EPOCHS = 10
LR = 1e-4
nom_grafica = input("Nom de la gràfica a wandb: ")

wandb.init(
    project="ciudades-resnet18",
    name=nom_grafica,
    config={ #indiquem els hiperparàmetres i altres detalls del projecte que volem trackejar a wandb
        "epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
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
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# =========================
# MODEL
# =========================

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

#Per a resnet18, el numero de in_features de la capa final és 512 sempre
model.fc = FCFinal(model.fc.in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) #Fem servir cross entropy loss pq estem en classificació multiclasse
#Sense el weight la funcio de loss seria L = -log(p_true_class), però amb el weight és L = - w_y * log(p_true_class) on w_y és el pes associat a la classe verdadera,
#El que fa és que si una classe és més rara (té menys exemples al train) li assigna un pes més alt, fent que els errors en aquesta classe siguin més importants per a la funció de loss i ajudant al model a aprendre millor aquesta classe minoritària
#Canvi important: class_weights.to(device) posa els pesos al mateix dispositiu que el model, evitant errors CPU/GPU.

optimizer = optim.Adam(model.parameters(), lr=LR)

#Especifiquem què mirem amb wandb, cada 10 batches guardem els gardients i guardem gradient i parmeteres cada 10 batches, així podem veure com evolucionen al llarg de l'entrenament
wandb.watch(model, criterion, log="all", log_freq=10)

#Guardem el millor model segons la validation accuracy
best_val_acc = 0.0

# Guardarem les prediccions de la millor validació per fer la seva matriu de confusió
best_val_preds = []
best_val_labels = []

# =========================
# TRAIN
# =========================

for epoch in range(NUM_EPOCHS):
    model.train() #indiquem què estem fent entrenament, però s'han d'ajustar els paràmetres tipo dropout o batchnorm

    train_loss_total = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader: #anem reorrentant els batches del train_loader

        #non_blocking=True pot accelerar la transferència CPU -> GPU quan pin_memory=True al DataLoader
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images) #fem el forward pass i obtenim les prediccions
        loss = criterion(outputs, labels) #calculem la loss

        optimizer.zero_grad() #posem els gradient a 0
        loss.backward() #calculem els gradients fent el backward pass
        optimizer.step() #actualitzem els pesos del model fent un pas d'optimització

        train_loss_total += loss.item()

        _, preds = torch.max(outputs, 1) #mirem quina classe ha predicho el model per cada imatge del batch
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

    # Guardem labels i prediccions de validació per poder fer matriu de confusió
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

            # Guardem les prediccions i labels en format llista per W&B
            current_val_preds.extend(preds.cpu().tolist())
            current_val_labels.extend(labels.cpu().tolist())

    val_loss = val_loss_total / len(val_loader)
    val_acc = 100 * val_correct / val_total

    #Canvi a wandb: noms agrupats perquè train i validation quedin junts a les gràfiques de loss i accuracy
    wandb.log({
        "epoch": epoch + 1,
        "loss/train": train_loss,
        "loss/validation": val_loss,
        "accuracy/train": train_acc,
        "accuracy/validation": val_acc
    })

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print("-" * 40)

    #Guardem el millor model, no només l'últim
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_preds = current_val_preds
        best_val_labels = current_val_labels

        torch.save(model.state_dict(), "best_resnet18_ciudades.pth")
        wandb.save("best_resnet18_ciudades.pth")
        print(f"Nou millor model guardat amb Val Acc: {best_val_acc:.2f}%")

# =========================
# TEST FINAL
# =========================

model.eval()

test_correct = 0
test_total = 0

# Guardem totes les prediccions i labels del test per construir la matriu de confusió
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

        # Guardem les prediccions i labels en format llista per W&B
        all_test_preds.extend(preds.cpu().tolist())
        all_test_labels.extend(labels.cpu().tolist())

test_acc = 100 * test_correct / test_total

wandb.log({
    "accuracy/test": test_acc,
    "accuracy/best_validation": best_val_acc,

    # Matriu de confusió de la millor validació
    # Serveix per veure quines classes confonia el model durant la validació
    "confusion_matrix/validation": wandb.plot.confusion_matrix(
        preds=best_val_preds,
        y_true=best_val_labels,
        class_names=class_names
    ),

    # Matriu de confusió del test
    # Serveix per veure els errors finals sobre dades no vistes
    "confusion_matrix/test": wandb.plot.confusion_matrix(
        preds=all_test_preds,
        y_true=all_test_labels,
        class_names=class_names
    )
})

print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Best Val Accuracy: {best_val_acc:.2f}%")

# Guardem resultats finals al resum de W&B
wandb.run.summary["test_accuracy"] = test_acc
wandb.run.summary["best_val_accuracy"] = best_val_acc

# =========================
# GUARDAR MODELO
# =========================

#Guardem també l'últim model entrenat
torch.save(model.state_dict(), "last_resnet18_ciudades.pth")
wandb.save("last_resnet18_ciudades.pth")

wandb.finish()