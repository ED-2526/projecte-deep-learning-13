
"""
import pickle
import torch
import matplotlib.pyplot as plt

# Carregar dades
with open("test_results.pkl", "rb") as f:
    data = pickle.load(f)

outputs = data["outputs"]
labels = data["labels"]

# Probabilitats
probs = torch.softmax(outputs, dim=1)

# Prediccions i confiança
max_probs, preds = torch.max(probs, dim=1)

# Correctes / incorrectes
correct_mask = preds == labels

correct_probs = max_probs[correct_mask]
incorrect_probs = max_probs[~correct_mask]

# Accuracy
accuracy = correct_mask.float().mean() * 100
print(f"Accuracy: {accuracy:.2f}%")

# Histograma
plt.figure(figsize=(10,6))


plt.hist(
    incorrect_probs.numpy(),
    bins=30,
    alpha=0.7,
    label="Incorrectes"
)

plt.xlabel("Probabilitat màxima predita")
plt.ylabel("Nombre d'exemples")
plt.title("Confiança del model: correctes vs incorrectes")

plt.legend()

plt.savefig("confidence_correct_vs_incorrect.png")
plt.show()
"""
import pickle
import torch
import matplotlib.pyplot as plt

# Carregar dades
with open("test_results.pkl", "rb") as f:
    data = pickle.load(f)

outputs = data["outputs"]
labels = data["labels"]

# -----------------------------------
# Probabilitats
# -----------------------------------

probs = torch.softmax(outputs, dim=1)

# Predicció i probabilitat màxima
max_probs, preds = torch.max(probs, dim=1)

# -----------------------------------
# Només errors
# -----------------------------------

incorrect_mask = preds != labels

incorrect_probs = max_probs[incorrect_mask]

incorrect_labels = labels[incorrect_mask]

# Probabilitat de la classe correcta
correct_class_probs = probs[
    incorrect_mask,
    incorrect_labels
]

# -----------------------------------
# Diferència
# -----------------------------------

difference = incorrect_probs - correct_class_probs

print("Mitjana diferència:",
      difference.mean().item())

print("Màxima diferència:",
      difference.max().item())

# -----------------------------------
# Histograma
# -----------------------------------

plt.figure(figsize=(10,6))

plt.hist(
    difference.numpy(),
    bins=30,
    alpha=0.8
)

plt.xlabel(
    "Prob(predicció incorrecta) - Prob(classe correcta)"
)

plt.ylabel("Nombre d'errors")

plt.title(
    "Diferència de confiança en errors"
)

plt.savefig("difference_wrong_vs_correct_probability.png")

plt.show()



"""
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataloaders import get_dataloaders

# Carregar dataloaders i noms de classes
train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()

# Carregar dades
with open("test_results.pkl", "rb") as f:
    data = pickle.load(f)

outputs = data["outputs"]
labels = data["labels"]

# Probabilitats
probs = torch.softmax(outputs, dim=1)

# Prediccions i confiança
max_probs, preds = torch.max(probs, dim=1)

# Incorrectes amb confiança > 0.9
mask = (preds != labels) & (max_probs > 0.9)

filtered_preds = preds[mask]
filtered_labels = labels[mask]

print(f"Nombre d'errors amb confiança > 0.9: {len(filtered_preds)}")

# IMPORTANT:
# assegurem que apareixen totes les classes
num_classes = len(class_names)

cm = confusion_matrix(
    filtered_labels.numpy(),
    filtered_preds.numpy(),
    labels=list(range(num_classes))
)

# Mostrar matriu
fig, ax = plt.subplots(figsize=(14,14))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format='d',
    xticks_rotation=90
)

plt.title("Errors amb confiança > 0.9")

plt.tight_layout()

plt.savefig("high_confidence_errors_confusion_matrix.png")
plt.show()

"""