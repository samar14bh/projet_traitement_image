"""
predictor.py
Charge les modèles .pth entraînés et expose une fonction predict_all()
qui renvoie les probabilités de chaque modèle sur une image donnée.
"""

import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from models import CustomCNN, VGGLight, ResNet18Adapted, MiniXception, EMOTIONS

IMG_SIZE  = 64
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Correspondance nom → classe + fichier .pth ────────────────────────────────
MODEL_REGISTRY = {
    "Custom CNN":       {"cls": CustomCNN,       "file": "model_custom_cnn.pth"},
    "VGG-Light":        {"cls": VGGLight,         "file": "finetuned_vgg_light_improved.pth"},
    "ResNet18-Adapted": {"cls": ResNet18Adapted,  "file": "model_resnet18_adapted.pth"},
    "Mini-Xception":    {"cls": MiniXception,     "file": "model_mini_xception.pth"},
}

# ── Statistiques enregistrées lors de l'entraînement ─────────────────────────
MODEL_STATS = {
    "Custom CNN":       {"val_acc": 57.95, "params": 2_192_327, "epochs": 30},
    "VGG-Light":        {"val_acc": 64.24, "params": 4_428_487, "epochs": 30},
    "ResNet18-Adapted": {"val_acc": 64.84, "params": 11_171_271,"epochs": 30},
    "Mini-Xception":    {"val_acc": 60.05, "params": 56_951,    "epochs": 26},
}

# ── Transformation identique à val_transform de l'entraînement ───────────────
_val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ── Prétraitement CLAHE (reproduit le pipeline du notebook) ──────────────────

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Reçoit une PIL Image (RGB ou L).
    Retourne un tensor [1, 1, 64, 64] prêt pour l'inférence.
    Applique : grayscale → resize 64×64 → CLAHE → normalisation [-1, 1].
    """
    img = np.array(pil_image.convert("L"))                          # grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    pil_gray = Image.fromarray(img)
    tensor = _val_transform(pil_gray).unsqueeze(0)                  # [1,1,64,64]
    return tensor


# ── Chargement des poids ──────────────────────────────────────────────────────

def load_model(name: str, models_dir: str = "saved_models") -> torch.nn.Module | None:
    """
    Charge le modèle <name> depuis models_dir/.
    Retourne None si le fichier .pth est introuvable.
    """
    info = MODEL_REGISTRY.get(name)
    if info is None:
        return None

    path = os.path.join(models_dir, info["file"])
    if not os.path.exists(path):
        return None

    model = info["cls"]()
    checkpoint = torch.load(path, map_location=DEVICE)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


# ── Inférence ────────────────────────────────────────────────────────────────

def predict_single(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Retourne un array numpy de probabilités (7,)."""
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    return probs


def predict_all(pil_image: Image.Image, models_dir: str = "saved_models") -> dict:
    """
    Lance l'inférence de tous les modèles disponibles sur pil_image.

    Retourne un dict :
    {
        "Custom CNN":       {"probs": array(7,), "predicted": "happy", "confidence": 0.82, "available": True},
        "VGG-Light":        {...},
        ...
        "_tensor":  tensor,   # tensor prétraité, pour affichage
    }
    """
    tensor = preprocess_image(pil_image)
    results = {"_tensor": tensor}

    for name in MODEL_REGISTRY:
        model = load_model(name, models_dir)
        if model is None:
            results[name] = {"available": False}
            continue

        probs      = predict_single(model, tensor)
        pred_idx   = int(np.argmax(probs))
        results[name] = {
            "available":   True,
            "probs":       probs,
            "predicted":   EMOTIONS[pred_idx],
            "confidence":  float(probs[pred_idx]),
        }

    return results
