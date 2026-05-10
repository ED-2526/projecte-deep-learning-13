import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import wandb
from dataloaders import get_dataloaders, BATCH_SIZE
from agrupa_continents import mapping_continents, to_continent_index_list
import wandb.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# =========================
# CONFIG
# =========================

NUM_EPOCHS = 10
LR = 1e-4
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
        "fc": "Linear(96)-BatchNorm-ReLU-Linear",
        "augmentation": "augmentation: RandomResizedCrop + RandomHorizontalFlip + ColorJitter"
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
# Batch Normalization:
# Normalitza la sortida de la capa Linear per a cada batch (mitjana ~0 i desviació ~1),
# estabilitzant els valors interns de la xarxa. Això fa que l'entrenament sigui més
# ràpid i estable, evita problemes de valors massa grans o petits, i millora la
# generalització del model. A més, incorpora dos paràmetres aprenables (gamma i beta)
# que permeten ajustar l'escala i el desplaçament de les dades normalitzades.

class FCFinal(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features,96),
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


#EFFICIENTNET B0:
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = FCFinal(in_features, num_classes)


model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) #Fem servir cross entropy loss pq estem en classificació multiclasse
#Sense el weight la funcio de loss seria L = -log(p_true_class), però amb el weight és L = - w_y * log(p_true_class) on w_y és el pes associat a la classe verdadera,
#El que fa és que si una classe és més rara (té menys exemples al train) li assigna un pes més alt, fent que els errors en aquesta classe siguin més importants per a la funció de loss i ajudant al model a aprendre millor aquesta classe minoritària
#Canvi important: class_weights.to(device) posa els pesos al mateix dispositiu que el model, evitant errors CPU/GPU.

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


#Especifiquem què mirem amb wandb, cada 10 batches guardem els gardients i parmeteres cada 10 batches, així podem veure com evolucionen al llarg de l'entrenament
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


    for i, (images, labels) in enumerate(train_loader):  # loop de batches 

        if (i + 1) % 200 == 0:  # print cada 200 batches per veure com va l'entrenament i la mida dels batches (l'últim batch pot ser més petit si no drop_last=True al DataLoader)
            print(f"Batch {i+1} | mida batch: {images.size(0)}")

        #non_blocking=True pot accelerar la transferència CPU -> GPU quan pin_memory=True al DataLoader
        images = images.to(device, non_blocking=True) #al carregar les imatges un proces no bloquegi a l'altre
        labels = labels.to(device, non_blocking=True)

        outputs = model(images) #fem el forward pass i obtenim les prediccions
        loss = criterion(outputs, labels) #calculem la loss

        #backpropagation i optimització
        optimizer.zero_grad() #posem els gradient a 0
        loss.backward() #calculem els gradients fent el backward pass
        optimizer.step() #actualitzem els pesos del model fent un pas d'optimització

        train_loss_total += loss.item()

        _, preds = torch.max(outputs, 1) #mirem quina classe ha predit el model per cada imatge del batch
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

    with torch.no_grad():  # no calcula gradients durant la validació
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

    # Canvi a wandb: noms agrupats perquè train i validation quedin junts a les gràfiques
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





# funció confussion matrix per W&B, amb opció de normalitzar per mostrar percentatges en lloc de nombres absoluts
def log_cm(y_true, y_pred, class_names, title, key, normalize=False):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    wandb.log({key: wandb.Image(plt)})
    plt.close()


wandb.log({
    "accuracy/test": test_acc,
    "accuracy/best_validation": best_val_acc,
})

# CIUTATS
log_cm(best_val_labels, best_val_preds, class_names,
       "Validació: Ciutats",
       "confusion_matrix/cities/validation")

log_cm(all_test_labels, all_test_preds, class_names,
       "Test: Ciutats",
       "confusion_matrix/cities/test")


# Imprimim també els resultats per consola})


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