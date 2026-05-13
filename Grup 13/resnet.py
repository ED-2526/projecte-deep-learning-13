import torch
import torch.nn as nn
from torchvision import models

# ==========================================
# DEFINICIÓ DE LA CAPÇALERA PERSONALITZADA
# ==========================================
# Aquesta classe substitueix l'última capa del model original per adaptar-lo 
# al nostre nombre de ciutats i aplicar les millores de generalització.

class FCFinal(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        # Creem una seqüència de capes (FCN amb 96 neurones)
        self.classifier = nn.Sequential(
            # Capa lineal que redueix de les features d'EfficientNet a 96
            nn.Linear(in_features, 96),
            
            # Batch Normalization:
            # Normalitza la sortida de la capa Linear per a cada batch (mitjana ~0 i desviació ~1),
            # estabilitzant els valors interns de la xarxa. Això fa que l'entrenament sigui més
            # ràpid i estable, evita problemes de valors massa grans o petits, i millora la
            # generalització del model. A més, incorpora dos paràmetres aprenables (gamma i beta)
            # que permeten ajustar l'escala i el desplaçament de les dades normalitzades.
            nn.BatchNorm1d(96),
            
            # Funció d'activació per afegir no-linealitat
            nn.ReLU(),
            
            # Dropout: Apaguem el 30% de les neurones aleatòriament durant el train 
            # per evitar que el model depengui de neurones concretes (redueix l'overfitting).
            nn.Dropout(p=0.3),
            
            # Capa final que projecta a les N ciutats que tenim al dataset
            nn.Linear(96, num_classes)
        )

    def forward(self, x):
        # El forward pass simplement passa les dades pel classificador definit a sobre
        return self.classifier(x)


# ==========================================
# CONSTRUCTOR DEL MODEL
# ==========================================

def get_model(num_classes):
    """
    Funció que construeix el model complet. 
    Es pot cridar des del main.py fent: model = get_model(len(class_names))
    """
    
    # EFFICIENTNET B0:
    # Carreguem el model preentrenat amb ImageNet. 
    # Tot i que l'script es digui resnet.py, fem servir EfficientNet-B0 per la seva eficiència.
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Obtenim el nombre de característiques d'entrada que genera EfficientNet abans de la seva capa final
    # En el cas de la B0, sol ser 1280.
    in_features = model.classifier[1].in_features
    
    # Substituïm el classificador original d'ImageNet (1000 classes) pel nostre FCFinal de 96 neurones.
    model.classifier[1] = FCFinal(in_features, num_classes)
    
    return model

