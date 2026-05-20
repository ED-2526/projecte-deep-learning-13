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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexSpecialistFC(nn.Module):
    
    def __init__(self, in_features, num_subclasses=3):
        super().__init__()
        self.classifier = nn.Sequential(
            # 1. Mantenim una bona part de la informació original (reduïm a 512 o 256)
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),            # Un dropout fort aquí protegeix del vostre dataset petit
            
            # 2. Ara sí, compactem cap a la decisió
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            # 3. Capa de sortida (les 3 o 4 ciutats)
            nn.Linear(256, num_subclasses)
        )
    
    def forward(self, x):
        return self.classifier(x)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight  # Aquí passarem els teus class_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculem el log_softmax
        log_p = F.log_softmax(inputs, dim=-1)
        
        # Cross-entropy amb els teus pesos per al càstig final
        ce_loss = F.nll_loss(log_p, targets, reduction='none', weight=self.weight)
        
        # Cross-entropy sense pesos per obtenir la probabilitat real (pt) de la classe correcta
        ce_loss_unweighted = F.nll_loss(log_p, targets, reduction='none')
        pt = torch.exp(-ce_loss_unweighted)
        
        # Apliquem el factor Focal Loss a la pèrdua pesada
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
       
        return focal_loss.mean()
        

# =========================
# CONFIG
# =========================

NUM_EPOCHS = 10

LR = 1e-4


PESOS_PATH = "pesos_bons_focal.pth"

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
        "loss": "FocalLoss",
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

criterion = FocalLoss(
    weight=class_weights.to(device), 
    gamma=2
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

# =====================================================================
# CONFIGURACIÓ DELS DOS ESPECIALISTES (USA i Europa)
# =====================================================================

# --- Especialista USA ---
usa_group_names = ['Boston', 'Chicago', 'Minneapolis']
usa_global_indices = [class_names.index(name) for name in usa_group_names]
global_to_local_usa = {global_idx: local_idx for local_idx, global_idx in enumerate(usa_global_indices)}
local_to_global_usa = {local_idx: global_idx for local_idx, global_idx in enumerate(usa_global_indices)}

especialista_usa = ComplexSpecialistFC(in_features=1280, num_subclasses=3).to(device)
optimizer_usa = optim.AdamW(especialista_usa.parameters(), lr=1e-3, weight_decay=1e-4)
criterion_usa = FocalLoss(gamma=2.0)

# --- Especialista Europa ---
euro_group_names = ['OSL', 'PRS', 'PRG', 'TRT']
euro_global_indices = [class_names.index(name) for name in euro_group_names]
global_to_local_euro = {global_idx: local_idx for local_idx, global_idx in enumerate(euro_global_indices)}
local_to_global_euro = {local_idx: global_idx for local_idx, global_idx in enumerate(euro_global_indices)}

especialista_euro = ComplexSpecialistFC(in_features=1280, num_subclasses=4).to(device)
optimizer_euro = optim.AdamW(especialista_euro.parameters(), lr=1e-3, weight_decay=1e-4)
criterion_euro = FocalLoss(gamma=2.0)

# --- Congelem la CNN base ---
for param in model.parameters():
    param.requires_grad = False
model.eval()

# --- Paràmetres globals del bucle ---
NUM_EPOCHS_ESP = 10
best_usa_val_acc = 0.0
best_euro_val_acc = 0.0

PESOS_USA_PATH = "especialista_usa_best.pth"
PESOS_EURO_PATH = "especialista_euro_best.pth"

print(f"\n=== Iniciant l'entrenament UNIFICAT dels Especialistes (10 èpoques) ===")

for epoch in range(NUM_EPOCHS_ESP):
    
    # -----------------------------------------------------------------
    # FASE DE TRAIN UNIFICADA
    # -----------------------------------------------------------------
    especialista_usa.train()
    especialista_euro.train()
    
    # Comptadors USA
    loss_tr_usa, corr_tr_usa, tot_tr_usa, bat_tr_usa = 0, 0, 0, 0
    # Comptadors Europa
    loss_tr_euro, corr_tr_euro, tot_tr_euro, bat_tr_euro = 0, 0, 0, 0
    
    for i, (images, labels) in enumerate(train_loader):
        if (i + 1) % 200 == 0:
            print(f"Batch {i+1} | mida batch: {images.size(0)}")
            
        images, labels = images.to(device), labels.to(device)
        
        # Extraiem característiques una sola vegada per a tot el batch (Estalvi brutal de temps)
        with torch.no_grad():
            batch_features = model.features(images)
            batch_features = model.avgpool(batch_features)
            batch_features = torch.flatten(batch_features, 1)
            
        # 🇺🇸 SUB-BUCLE USA
        mask_usa = torch.isin(labels, torch.tensor(usa_global_indices).to(device)) #comparem cada etiqueta del batch per agafar només les que pertanyen a USA
        if mask_usa.sum() > 1: #necessitem més d'una mostra per fer BatchNorm
            bat_tr_usa += 1 #comptador de batches efectius per a USA
            feat_usa = batch_features[mask_usa] #agafem només les característiques de les mostres d'USA
            lbl_global_usa = labels[mask_usa] #agafem les etiquetes globals d'aquestes mostres (0, 3, 5)
            lbl_local_usa = torch.tensor([global_to_local_usa[lbl.item()] for lbl in lbl_global_usa]).to(device) #passa dels noms globals (0, 3, 5) a locals (0, 1, 2) per a l'especialista USA
            
            out_usa = especialista_usa(feat_usa)
            loss_u = criterion_usa(out_usa, lbl_local_usa)
            
            optimizer_usa.zero_grad()
            loss_u.backward()
            optimizer_usa.step()
            
            loss_tr_usa += loss_u.item()
            _, preds_u = torch.max(out_usa, 1)
            tot_tr_usa += lbl_local_usa.size(0)
            corr_tr_usa += (preds_u == lbl_local_usa).sum().item()
            
        # 🇪🇺 SUB-BUCLE EUROPA
        mask_euro = torch.isin(labels, torch.tensor(euro_global_indices).to(device))
        if mask_euro.sum() > 1: # Protecció BatchNorm
            bat_tr_euro += 1
            feat_euro = batch_features[mask_euro]
            lbl_global_euro = labels[mask_euro]
            lbl_local_euro = torch.tensor([global_to_local_euro[lbl.item()] for lbl in lbl_global_euro]).to(device)
            
            out_euro = especialista_euro(feat_euro)
            loss_e = criterion_euro(out_euro, lbl_local_euro)
            
            optimizer_euro.zero_grad()
            loss_e.backward()
            optimizer_euro.step()
            
            loss_tr_euro += loss_e.item()
            _, preds_e = torch.max(out_euro, 1)
            tot_tr_euro += lbl_local_euro.size(0)
            corr_tr_euro += (preds_e == lbl_local_euro).sum().item()

    # Mitjanes de Train
    acc_tr_usa = (100 * corr_tr_usa / tot_tr_usa) if tot_tr_usa > 0 else 0
    loss_tr_usa = (loss_tr_usa / bat_tr_usa) if bat_tr_usa > 0 else 0
    acc_tr_euro = (100 * corr_tr_euro / tot_tr_euro) if tot_tr_euro > 0 else 0
    loss_tr_euro = (loss_tr_euro / bat_tr_euro) if bat_tr_euro > 0 else 0

    # -----------------------------------------------------------------
    # FASE DE VALIDACIÓ UNIFICADA
    # -----------------------------------------------------------------
    especialista_usa.eval()
    especialista_euro.eval()
    
    loss_val_usa, corr_val_usa, tot_val_usa, bat_val_usa = 0, 0, 0, 0
    loss_val_euro, corr_val_euro, tot_val_euro, bat_val_euro = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            batch_features = model.features(images)
            batch_features = model.avgpool(batch_features)
            batch_features = torch.flatten(batch_features, 1)
            
            # Validació USA
            mask_usa = torch.isin(labels, torch.tensor(usa_global_indices).to(device))
            if mask_usa.any():
                bat_val_usa += 1
                feat_usa = batch_features[mask_usa]
                lbl_local_usa = torch.tensor([global_to_local_usa[lbl.item()] for lbl in labels[mask_usa]]).to(device)
                out_usa = especialista_usa(feat_usa)
                loss_val_usa += criterion_usa(out_usa, lbl_local_usa).item()
                _, preds_u = torch.max(out_usa, 1)
                tot_val_usa += lbl_local_usa.size(0)
                corr_val_usa += (preds_u == lbl_local_usa).sum().item()
                
            # Validació Europa
            mask_euro = torch.isin(labels, torch.tensor(euro_global_indices).to(device))
            if mask_euro.any():
                bat_val_euro += 1
                feat_euro = batch_features[mask_euro]
                lbl_local_euro = torch.tensor([global_to_local_euro[lbl.item()] for lbl in labels[mask_euro]]).to(device)
                out_euro = especialista_euro(feat_euro)
                loss_val_euro += criterion_euro(out_euro, lbl_local_euro).item()
                _, preds_e = torch.max(out_euro, 1)
                tot_val_euro += lbl_local_euro.size(0)
                corr_val_euro += (preds_e == lbl_local_euro).sum().item()

    # Mitjanes de Validació
    acc_val_usa = (100 * corr_val_usa / tot_val_usa) if tot_val_usa > 0 else 0
    loss_val_usa = (loss_val_usa / bat_val_usa) if bat_val_usa > 0 else 0
    acc_val_euro = (100 * corr_val_euro / tot_val_euro) if tot_val_euro > 0 else 0
    loss_val_euro = (loss_val_euro / bat_val_euro) if bat_val_euro > 0 else 0

    # -----------------------------------------------------------------
    # LOGS COMPLETS A W&B
    # -----------------------------------------------------------------
    wandb.log({
        "specialists_combined_epoch": epoch + 1,
        # Gràfiques USA
        "usa_specialist/loss_train": loss_tr_usa,
        "usa_specialist/loss_validation": loss_val_usa,
        "usa_specialist/accuracy_train": acc_tr_usa,
        "usa_specialist/accuracy_validation": acc_val_usa,
        # Gràfiques Europa
        "euro_specialist/loss_train": loss_tr_euro,
        "euro_specialist/loss_validation": loss_val_euro,
        "euro_specialist/accuracy_train": acc_tr_euro,
        "euro_specialist/accuracy_validation": acc_val_euro,
    })

    # Prints de pantalla resumits per època
    print(f"\n=================== ÈPOCA {epoch+1}/{NUM_EPOCHS_ESP} ===================")
    print(f"🇺🇸 USA  | Train Acc: {acc_tr_usa:.2f}% (Loss: {loss_tr_usa:.4f}) | Val Acc: {acc_val_usa:.2f}% (Loss: {loss_val_usa:.4f})")
    print(f"🇪🇺 EURO | Train Acc: {acc_tr_euro:.2f}% (Loss: {loss_tr_euro:.4f}) | Val Acc: {acc_val_euro:.2f}% (Loss: {loss_val_euro:.4f})")

    # Guardat independent del millor model de cada especialitat
    if acc_val_usa > best_usa_val_acc:
        best_usa_val_acc = acc_val_acc = acc_val_usa
        torch.save(especialista_usa.state_dict(), PESOS_USA_PATH)
        print(f"  --> 🎉 Nou millor especialista USA guardat! ({best_usa_val_acc:.2f}%)")
        
    if acc_val_euro > best_euro_val_acc:
        best_euro_val_acc = acc_val_euro
        torch.save(especialista_euro.state_dict(), PESOS_EURO_PATH)
        print(f"  --> 🎉 Nou millor especialista EURO guardat! ({best_euro_val_acc:.2f}%)")
    print("=" * 46)

# Carreguem els pesos òptims de tots dos abans d'anar al TEST FINAL
especialista_usa.load_state_dict(torch.load(PESOS_USA_PATH))
especialista_euro.load_state_dict(torch.load(PESOS_EURO_PATH))
especialista_usa.eval()
especialista_euro.eval()
print("\nTots dos especialistes carregats amb el seu millor estat històric. Llestos pel Test Final.")



# =========================
# TEST FINAL
# =========================

import pickle

# 🔴 CORREGIT: Posem TOTS els models en mode eval a dalt de tot
model.eval()
especialista_usa.eval()
especialista_euro.eval() 

all_final_preds = []
all_final_labels = []

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. El model general fa la seva predicció primària
        outputs_general = model(images)
        _, preds_general = torch.max(outputs_general, 1)
        
        for idx in range(images.size(0)):
            pred_inicial = preds_general[idx].item()
            label_real = labels[idx].item()
            
            # 2. RÚTER TRIPLE:
            
            # CAMÍ A: Si la predicció cau en el pou d'USA, activem l'especialista USA
            if pred_inicial in usa_global_indices:
                # Extraiem característiques d'aquesta imatge concreta
                feat = model.features(images[idx].unsqueeze(0))
                feat = model.avgpool(feat)
                feat = torch.flatten(feat, 1)
                
                # L'especialista dicta sentència (retorna 0, 1, o 2)
                output_esp = especialista_usa(feat)
                _, pred_local = torch.max(output_esp, 1)
                
                # 🔴 CORREGIT: Utilitzem el nom correcte del diccionari unificat (local_to_global_usa)
                pred_final = local_to_global_usa[pred_local.item()]
                
            # CAMÍ B: Pertany al pou de confusió d'Europa
            elif pred_inicial in euro_global_indices:
                feat = model.features(images[idx].unsqueeze(0))
                feat = model.avgpool(feat)
                feat = torch.flatten(feat, 1)
                
                output_esp = especialista_euro(feat)
                _, pred_local = torch.max(output_esp, 1)
                pred_final = local_to_global_euro[pred_local.item()] 
            
            # CAMÍ C: Si la predicció és neta, ens en fiem completament
            else:
                pred_final = pred_inicial
                
            all_final_preds.append(pred_final)
            all_final_labels.append(label_real)
            
            # Actualitzem el total i comprovem si ha encertat el sistema combinat
            test_total += 1
            if pred_final == label_real:
                test_correct += 1

# Càlcul de l'accuracy final del sistema jeràrquic complet
test_acc = 100 * test_correct / test_total
print(f"\n🚀 Test accuracy del sistema combinat: {test_acc:.2f}%")

"""
# Concatenem tensors
all_outputs = torch.cat(all_outputs, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# Guardem resultats
with open("test_results.pkl", "wb") as f:
    pickle.dump({
        "outputs": all_outputs,
        "labels": all_labels,
        "predictions": all_test_preds,
        "accuracy": test_acc
    }, f)

"""
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

"""
import random   

print("\n===== GRADCAM OSLO > 0.9 =====")

# -----------------------------------
# Probabilitats
# -----------------------------------

probs = torch.softmax(all_outputs, dim=1)

max_probs, preds = torch.max(probs, dim=1)

# -----------------------------------
# Classe OSLO
# -----------------------------------

oslo_idx = class_names.index("OSL")

# -----------------------------------
# Trobar exemples
# -----------------------------------

wrong_oslo = []
correct_oslo = []

for i in range(len(all_labels)):

    real_label = all_labels[i].item()
    pred_label = preds[i].item()
    confidence = max_probs[i].item()

    # Només confiança > 0.9
    if pred_label == oslo_idx and confidence > 0.7:

        # Incorrectes
        if real_label != oslo_idx:
            wrong_oslo.append(i)

        # Correctes
        else:
            correct_oslo.append(i)

print(f"OSLO incorrectes >0.7: {len(wrong_oslo)}")
print(f"OSLO correctes >0.7: {len(correct_oslo)}")

# -----------------------------------
# Selecció aleatòria
# -----------------------------------

NUM_EXAMPLES = 5

wrong_oslo = random.sample(
    wrong_oslo,
    min(NUM_EXAMPLES, len(wrong_oslo))
)

correct_oslo = random.sample(
    correct_oslo,
    min(NUM_EXAMPLES, len(correct_oslo))
)

selected_indices = wrong_oslo + correct_oslo

# -----------------------------------
# Recuperar imatges
# -----------------------------------

selected_images = {}

current_idx = 0

for images, labels in test_loader:

    batch_size = images.size(0)

    for j in range(batch_size):

        global_idx = current_idx + j

        if global_idx in selected_indices:
            selected_images[global_idx] = images[j]

    current_idx += batch_size

# -----------------------------------
# GradCAM
# -----------------------------------

target_layers = [model.features[-1]]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

# -----------------------------------
# Plot
# -----------------------------------

fig, axes = plt.subplots(
    2,
    NUM_EXAMPLES,
    figsize=(4 * NUM_EXAMPLES, 8)
)

# =========================
# INCORRECTES
# =========================

for col, idx in enumerate(wrong_oslo):

    image_tensor = selected_images[idx].unsqueeze(0).to(device)

    targets = [ClassifierOutputTarget(oslo_idx)]

    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets
    )[0]

    img = selected_images[idx].permute(1, 2, 0).cpu().numpy()

    img = img - img.min()
    img = img / (img.max() + 1e-8)

    visualization = show_cam_on_image(
        img,
        grayscale_cam,
        use_rgb=True
    )

    axes[0, col].imshow(visualization)

    axes[0, col].set_title(
        f"REAL: {class_names[all_labels[idx]]}\n"
        f"PRED: OSL\n"
        f"{max_probs[idx].item():.2f}"
    )

    axes[0, col].axis("off")

# =========================
# CORRECTES
# =========================

for col, idx in enumerate(correct_oslo):

    image_tensor = selected_images[idx].unsqueeze(0).to(device)

    targets = [ClassifierOutputTarget(oslo_idx)]

    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets
    )[0]

    img = selected_images[idx].permute(1, 2, 0).cpu().numpy()

    img = img - img.min()
    img = img / (img.max() + 1e-8)

    visualization = show_cam_on_image(
        img,
        grayscale_cam,
        use_rgb=True
    )

    axes[1, col].imshow(visualization)

    axes[1, col].set_title(
        f"REAL: OSL\n"
        f"PRED: OSL\n"
        f"{max_probs[idx].item():.2f}"
    )

    axes[1, col].axis("off")

# Labels files
axes[0, 0].set_ylabel("Incorrectes", fontsize=16)
axes[1, 0].set_ylabel("Correctes", fontsize=16)

plt.tight_layout()

plt.savefig("gradcam_oslo_confidence_0_7_focal_loss.png")

plt.show()
"""

# =========================
# LOGS
# =========================

# Enviem les últimes mètriques del sistema combinat a WandB
wandb.log({
    "accuracy/test": test_acc,
    "accuracy/best_validation": best_val_acc,
})

# 🔴 CORREGIT: Li passem les llistes 'final' perquè ens pinti la matriu 
# amb els errors de Chicago, Praga, etc. ja corregits pels especialistes!
log_cm(
    all_final_labels,
    all_final_preds,
    class_names,
    "Test: Ciutats (Sistema Combinat)",
    "confusion_matrix/cities/test"
)

print(f"Test Accuracy Final: {test_acc:.2f}%")

if not skip_training:
    print(f"Best Val Accuracy (Model General): {best_val_acc:.2f}%")

# Guardem el resum definitiu al panell de W&B
wandb.run.summary["test_accuracy"] = test_acc
wandb.run.summary["augmentation"] = AUGMENTATION_TYPE

if not skip_training:
    wandb.run.summary["best_val_accuracy"] = best_val_acc

# Guardem l'estat final de la xarxa base per seguretat
torch.save(model.state_dict(), "last_efficientnet_b0_ciudades.pth")
wandb.save("last_efficientnet_b0_ciudades.pth")

# Tanquem la sessió de Weights & Biases de manera neta
wandb.finish()