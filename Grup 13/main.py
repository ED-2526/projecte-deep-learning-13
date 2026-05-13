import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Importem els nostres mòduls modulars
from dataloaders import get_dataloaders
from resnet import get_model
from dubtes import CityCostLoss

# =========================
# CONFIGURACIÓ GLOBAL
# =========================
NUM_EPOCHS_BASE = 10    
NUM_EPOCHS_REFORC = 5   
LR_BASE = 1e-4          
LR_REFORC = 1e-5        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_confusion_matrix(model, loader, class_names, title, wandb_key):
    """
    Funció per generar la matriu de confusió i pujar-la a W&B.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Realitat')
    plt.xlabel('Predicció')
    
    wandb.log({wandb_key: wandb.Image(plt)})
    plt.close()

def main():
    # Inicialització del projecte a WandB
    nom_grafica = input("Nom de l'experiment a W&B: ")
    wandb.init(project="ciudades-modular-reforc", name=nom_grafica)

    # 1. CARREGAR DADES
    # El Pickle ens garanteix que train/val/test són sempre els mateixos grups d'imatges.
    print("Carregant dades...")
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()
    class_weights = class_weights.to(DEVICE)
    num_classes = len(class_names)

    # 2. INICIALITZAR MODEL
    # Creem la EfficientNet-B0 definida a resnet.py
    print(f"Configurant model a {DEVICE}...")
    model = get_model(num_classes).to(DEVICE)
    
    # --- FASE 1: ENTRENAMENT BASE ---
    print("\n>>> FASE 1: Iniciant entrenament base (10 epochs)")
    criterion_base = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS_BASE):
        # --- SUB-PART: EVALUACIÓ DEL TRAIN (ENTRENAMENT) ---
        model.train() # Activem Dropout i Batch Normalization per a entrenament
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            # Enviem les dades del batch a la GPU
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            # Resetegem els gradients de l'optimitzador (obligatori abans de cada batch)
            optimizer.zero_grad()
            
            # FORWARD PASS: Passem les imatges pel model per obtenir les prediccions (logits)
            outputs = model(images)
            
            # CÀLCUL DE LA LOSS: Comparem el que diu el model amb la realitat
            loss = criterion_base(outputs, labels)
            
            # BACKWARD PASS: Calculem els gradients de la pèrdua respecte als pesos
            loss.backward()
            
            # ACTUALITZACIÓ: L'optimitzador (AdamW) mou els pesos en la direcció que redueix l'error
            optimizer.step()
            
            # ACUMULACIÓ DE MÈTRIQUES: Sumem la loss i calculem quantes imatges hem encertat
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1) # La classe predita és la que té el valor més alt
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        # --- SUB-PART: EVALUACIÓ DEL VALIDATION (VALIDACIÓ) ---
        model.eval() # Congelem Dropout i Batch Normalization per a una avaluació estable
        val_correct, val_total = 0, 0
        
        # Desactivem el càlcul de gradients: estalvia memòria i accelera el procés
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # Forward pass: obtenim les prediccions amb les dades de validació
                outputs = model(images)
                
                # Calculem l'accuracy: comparem la predicció més alta amb l'etiqueta real
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        
        # Càlcul de percentatge final d'encert en validació
        val_acc = 100 * val_correct / val_total
        
        # Enviem les dades de l'epoch a la plataforma WandB
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss / len(train_loader),
            "acc/val": val_acc,
            "fase": 1
        })
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS_BASE} | Val Acc: {val_acc:.2f}%")

        # LOGICA DEL CHECKPOINT: Si hem millorat la validació, guardem l'estat dels pesos
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_base.pth")

    # Matriu de confusió per analitzar els errors (Chicago vs Boston/Minne) abans del reforç
    log_confusion_matrix(model, val_loader, class_names, "Confusió Fase Base", "cm_fase_base")

    # --- FASE 2: REFORÇ AMB MATRIU DE COSTOS ---
    print("\n>>> FASE 2: Aplicant CityCostLoss (reforç dirigit)")
    # Carreguem els pesos del millor moment de la fase base
    model.load_state_dict(torch.load("best_model_base.pth"))
    
    # Canviem la funció de pèrdua a la CityCostLoss de dubtes.py (penalització x7)
    criterion_reforc = CityCostLoss(class_names, base_weights=class_weights, penalty=7.0)
    # Baixem el Learning Rate perquè el model només faci "retocs quirúrgics"
    optimizer_reforc = optim.AdamW(model.parameters(), lr=LR_REFORC)

    for epoch in range(NUM_EPOCHS_REFORC):
        model.train() # Tornem a mode entrenament per a la nova fase
        r_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_reforc.zero_grad()
            
            # Aquí la Loss és diferent: penalitza més fortament si es confon amb Chicago
            outputs = model(images)
            loss = criterion_reforc(outputs, labels)
            
            loss.backward()
            optimizer_reforc.step()
            r_loss += loss.item()

        # Validació fase reforç (repetim la lògica anterior per veure l'evolució de l'accuracy)
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        val_acc = 100 * val_correct / len(val_loader.dataset)
        
        wandb.log({
            "epoch": NUM_EPOCHS_BASE + epoch + 1,
            "loss/reforc": r_loss / len(train_loader),
            "acc/val": val_acc,
            "fase": 2
        })
        print(f"Reforç Epoch {epoch+1}/{NUM_EPOCHS_REFORC} | Val Acc: {val_acc:.2f}%")

    # Generem la matriu final per comprovar si hem corregit els dubtes detectats
    log_confusion_matrix(model, val_loader, class_names, "Confusió Final Post-Reforç", "cm_final")

    # --- TEST FINAL ---
    # Avaluació única sobre dades que el model mai ha vist (ni en train ni en val)
    print("\n>>> TEST FINAL amb model reforçat")
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
    
    test_acc = 100 * test_correct / len(test_loader.dataset)
    print(f"Accuracy Final en Test: {test_acc:.2f}%")
    wandb.run.summary["final_test_acc"] = test_acc

    torch.save(model.state_dict(), "last_model_final.pth")
    wandb.finish()

if __name__ == "__main__":
    main()