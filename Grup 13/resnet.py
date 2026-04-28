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
    config={ #indiquem els hiperparàmetres i altres detalls del projecte que volem trackejar a wandb
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

criterion = nn.CrossEntropyLoss(weight=class_weights) #Fem servir cross entropy loss pq estem en classificació multiclasse
#Sense el weight la funcio de loss seria L = -log(p_true_class), però amb el weight és L = - w_y * log(p_true_class) on w_y és el pes associat a la classe verdadera, 
#El que fa és que si una classe és més rara (té menys exemples al train) li assigna un pes més alt, fent que els errors en aquesta classe siguin més importants per a la funció de loss i ajudant al model a aprendre millor aquesta classe minoritària
optimizer = optim.Adam(model.parameters(), lr=LR)

#Especifiquem què mirem amb wandb, cada 10 batches guardem els gardients i guardem gradient i parmeteres cada 10 batches, així podem veure com evolucionen al llarg de l'entrenament
wandb.watch(model, criterion, log="all", log_freq=10)

# =========================
# TRAIN
# =========================

for epoch in range(NUM_EPOCHS):
    model.train() #indiquem què estem fent entrenament, però s'han d'ajustar els paràmetres tipo dropout o batchnorm

    train_loss_total = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader: #anem reorrentant els batches del train_loader

        print("Nou batch d'entrenament:", images.size(0))

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images) #fem el forward pass i obtenim les prediccions
        loss = criterion(outputs, labels) #calculem la loss 

        optimizer.zero_grad() #posem els gradient a 0
        loss.backward() #calculem els gradients fent el backward pass
        optimizer.step() #actualitzem els pesos del model fent un pas d'optimització

        train_loss_total += loss.item()

        _, preds = torch.max(outputs, 1) #mirem quina classe ha predicho el model per cada imatge del batch, torch.max retorna el valor máximo y su índice, al poner 1 le decimos que mire por filas, así que nos devuelve el índice de la clase con mayor probabilidad para cada imagen del batch
        train_total += labels.size(0) #suma que el número total de imágenes que hemos visto, que es el tamaño del batch (labels.size(0))
        train_correct += (preds == labels).sum().item() #comparamos las predicciones con las etiquetas verdaderas (preds == labels) y sumamos el número de aciertos, (preds == labels) nos devuelve un tensor booleano donde cada posición es True si la predicción es correcta y False si no, al hacer .sum() contamos cuántos True hay, que es el número de aciertos, y con .item() convertimos ese número a un valor escalar de Python

    train_loss = train_loss_total / len(train_loader) #mitjana de loss per batch
    train_acc = 100 * train_correct / train_total #calculem el percentatge d'encerts

    # =========================
    # VALIDATION
    # =========================

    model.eval() #indiquem què estem fent evaluació, aquí volem comportament estable no cal ajustar paràmetres ni res 

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
