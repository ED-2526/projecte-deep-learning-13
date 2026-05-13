import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# FUNCIÓ DE PÈRDUA PERSONALITZADA (COSTOS)
# ==========================================

class CityCostLoss(nn.Module):
    """
    Aquesta classe implementa una CrossEntropyLoss modificada. 
    L'objectiu és penalitzar fortament els errors on el model prediu 'Chicago' 
    però la realitat és 'Boston' o 'Minneapolis'.
    """
    def __init__(self, class_names, base_weights=None, penalty=7.0):
        super(CityCostLoss, self).__init__()
        self.class_names = class_names
        self.base_weights = base_weights # Pesos originals per desbalanç de classes
        self.penalty = penalty
        
        # Identifiquem els índexs numèrics de les ciutats per a la lògica de la màscara
        try:
            self.idx_chicago = class_names.index("Chicago")
            self.idx_boston = class_names.index("Boston")
            self.idx_minneapolis = class_names.index("Minneapolis")
        except ValueError as e:
            print(f"Error: Una de les ciutats no s'ha trobat a la llista de classes: {e}")

    def forward(self, outputs, targets):
        """
        outputs: prediccions del model (logits)
        targets: etiquetes reals (ground truth)
        """
        # 1. Calculem la CrossEntropy estàndard batch a batch (sense fer la mitjana encara)
        # Fem servir 'reduction=none' per poder aplicar la penalització individualment
        base_loss = F.cross_entropy(outputs, targets, weight=self.base_weights, reduction='none')
        
        # 2. Obtenim quina és la predicció més probable que ha fet el model ara mateix
        preds = torch.argmax(outputs, dim=1)
        
        # 3. Creem una màscara booleana per identificar els errors específics:
        # Condició: (Realitat és Boston O Minneapolis) I (Predicció és Chicago)
        mask_boston_error = (targets == self.idx_boston) & (preds == self.idx_chicago)
        mask_minne_error = (targets == self.idx_minneapolis) & (preds == self.idx_chicago)
        
        full_mask = mask_boston_error | mask_minne_error
        
        # 4. Apliquem el multiplicador de penalització als exemples que compleixen la màscara
        # Això farà que el gradient sigui molt més gran i el model aprengui més ràpid a no fer-ho
        base_loss[full_mask] = base_loss[full_mask] * self.penalty
        
        # 5. Retornem la mitjana de la loss modificada per a tot el batch
        return base_loss.mean()

# ==========================================
# FUTURES IMPLEMENTACIONS (Classificador 4 ciutats)
# ==========================================

# Aquí és on anirà el classificador especialitzat per a Oslo, Praga, Toronto i París 
# que heu comentat. Per ara, ens centrem en la matriu de costos per a Chicago.