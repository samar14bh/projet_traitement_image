# FER2013 — Interface de Reconnaissance des Émotions Faciales

Web app **Streamlit** permettant de tester et comparer les 4 modèles CNN entraînés
sur le dataset FER2013 (7 classes d'émotions).

## Structure du projet

```
fer2013_app/
├── app.py              # Interface Streamlit principale
├── models.py           # Définitions PyTorch des 4 architectures
├── predictor.py        # Chargement des .pth + pipeline d'inférence
├── requirements.txt    # Dépendances Python
└── saved_models/       # ← Copiez vos fichiers .pth ici
    ├── model_custom_cnn.pth
    ├── model_vgg_light.pth
    ├── model_resnet18_adapted.pth
    └── model_mini_xception.pth
```

## Installation

```bash
# 1. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Copier les modèles entraînés
mkdir saved_models
cp /chemin/vers/vos/modeles/*.pth saved_models/
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur http://localhost:8501

## Fonctionnalités

| Fonctionnalité | Détail |
|---|---|
| Upload image | JPG, PNG, BMP, WEBP |
| Prétraitement | Grayscale → resize 64×64 → CLAHE (identique au notebook) |
| Inférence | 4 modèles en parallèle |
| Consensus | Vote majoritaire affiché en haut |
| Onglet comparaison | Bar chart groupé + radar + heatmap (Plotly) |
| Onglet par modèle | Top-K barres horizontales + méta informations |
| Seuil de confiance | Configurable dans la sidebar (défaut 30%) |
| Vérification .pth | La sidebar indique ✅/❌ pour chaque modèle |

## Modèles supportés

| Modèle | Fichier .pth | Val Acc |
|---|---|---|
| Custom CNN | model_custom_cnn.pth | 57.95% |
| VGG-Light | model_vgg_light.pth | 64.24% |
| ResNet18-Adapted | model_resnet18_adapted.pth | 64.84% |
| Mini-Xception | model_mini_xception.pth | 60.05% |

> L'application fonctionne même si certains .pth sont manquants —
> seuls les modèles disponibles sont utilisés.
