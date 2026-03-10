# LSTM-MLE

Pipeline de prévision de séries temporelles financières par apprentissage profond.

## Vue d'ensemble

Ce projet implémente un pipeline complet et modulaire pour la **prévision du rendement financier au pas de temps t+1** à partir de séries temporelles de prix. Il combine :

- **Log-rendements** comme signal stationnaire de base.
- **Transformée de Fourier Dynamique (DFT)** avec tapering et lissage spectral pour extraire les fréquences dominantes du marché.
- **Transformée en Ondelettes Discrète (DWT)** pour capturer les comportements basse et haute fréquence à différentes échelles temporelles.
- Un **réseau de neurones LSTM** (Long Short-Term Memory) pour apprendre les dépendances temporelles longues.

---

## Contexte mathématique

### 1. Log-rendements

Les prix financiers étant non-stationnaires, on les transforme en **log-rendements** :

```
r_t = ln(P_t / P_{t-1})
```

Cette transformation rend la série approximativement stationnaire et normalise les variations de prix quelle que soit leur échelle.

### 2. Transformée de Fourier Dynamique (DFT)

L'analyse spectrale sur fenêtres glissantes permet de détecter les **fréquences cycliques dominantes** du marché (saisonnalités, effets de calendrier, cycles économiques).

#### a) Cosine Bell Taper (Fenêtre de Tukey)

Pour réduire les **fuites spectrales** (*spectral leakage*) dues aux discontinuités aux bords des fenêtres, on applique une fenêtre de Tukey :

```
w(t) = { 0.5 * (1 - cos(2π t / (α N)))       pour  0 ≤ t < αN/2
        { 1                                     pour  αN/2 ≤ t < N(1 - α/2)
        { 0.5 * (1 - cos(2π (N-t) / (α N)))   pour  N(1 - α/2) ≤ t ≤ N
```

où `α` contrôle la fraction de la fenêtre couverte par les transitions cosinus.

#### b) Densité Spectrale de Puissance (DSP)

```
PSD(f) = |FFT(w · r)|²
```

#### c) Noyau de Daniell

Le spectre est lissé par une **moyenne mobile uniforme** (noyau rectangulaire) pour réduire la variance des estimations spectrales :

```
PSD_lissé(f) = (1 / K) Σ_{k=-K/2}^{K/2} PSD(f + k)
```

### 3. Transformée en Ondelettes Discrète (DWT)

La DWT décompose le signal en composantes multi-échelles :

- **Coefficients d'approximation (cA)** : comportement basse fréquence, tendance du marché.
- **Coefficients de détail (cD₁, cD₂, ..., cDₙ)** : comportement haute fréquence, sauts soudains, chocs de marché à différentes résolutions temporelles.

Pour chaque niveau, trois statistiques récapitulatives sont calculées :
- **Énergie** : `E = Σ cₖ²`
- **Valeur absolue moyenne (MAV)** : `MAV = (1/N) Σ |cₖ|`
- **Écart-type** : `σ = std(cₖ)`

### 4. Architecture LSTM

```
Entrée : (batch_size, seq_length, input_size)
    ↓
LSTM multi-couches (num_layers, hidden_size, dropout)
    ↓
Dernier état caché : (batch_size, hidden_size)
    ↓
Linear(hidden_size → 64) → ReLU → Dropout → Linear(64 → 1)
    ↓
Sortie : rendement prédit r_{t+1}
```

---

## Structure du projet

```
LSTM-MLE/
├── requirements.txt          # Dépendances Python
├── README.md                 # Documentation (ce fichier)
├── main.py                   # Script d'orchestration principal
├── data/
│   ├── __init__.py
│   └── returns.py            # Calcul des log-rendements
├── features/
│   ├── __init__.py
│   ├── fourier.py            # DFT dynamique : tapering + noyau de Daniell
│   └── wavelets.py           # DWT via PyWavelets : approximation + détails
├── models/
│   ├── __init__.py
│   └── lstm.py               # Module StockLSTM PyTorch
├── pipeline/
│   ├── __init__.py
│   ├── dataset.py            # Construction du jeu de données (fenêtres glissantes)
│   └── train.py              # Boucle d'entraînement (MSE, Adam, early stopping)
```

---

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip

### Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Utilisation

### Lancer le pipeline avec données synthétiques

```bash
python main.py
```

Ceci génère automatiquement une série de prix synthétiques par **Mouvement Brownien Géométrique** et exécute le pipeline complet.

### Options de la ligne de commande

```bash
python main.py [OPTIONS]

Options :
  --csv_path PATH       Chemin vers un fichier CSV de prix (colonne 'close')
  --epochs INT          Nombre maximum d'époques (défaut : 50)
  --seq_length INT      Longueur des séquences d'entrée (défaut : 60)
  --batch_size INT      Taille des mini-lots (défaut : 32)
  --hidden_size INT     Unités cachées LSTM (défaut : 128)
  --lr FLOAT            Taux d'apprentissage (défaut : 0.001)
  --num_layers INT      Couches LSTM empilées (défaut : 2)
  --dropout FLOAT       Probabilité de dropout (défaut : 0.2)
  --fourier_window INT  Taille de fenêtre Fourier (défaut : 64)
  --fourier_top_k INT   Nombre de fréquences Fourier (défaut : 5)
  --wavelet STR         Famille d'ondelettes (défaut : 'db4')
  --wavelet_level INT   Niveaux de décomposition DWT (défaut : 3)
  --patience INT        Patience early stopping (défaut : 15)
  --save_plot PATH      Chemin de sauvegarde de la courbe de perte
  --device STR          Dispositif ('cpu' ou 'cuda')
```

### Exemples

```bash
# Avec un fichier CSV et 100 époques
python main.py --csv_path donnees.csv --epochs 100

# Avec des séquences plus longues et un modèle plus grand
python main.py --seq_length 120 --hidden_size 256 --num_layers 3

# Sauvegarde de la courbe de perte
python main.py --epochs 50 --save_plot courbe_perte.png
```

---

## Exemple de sortie

```
Dispositif utilisé : cpu

[1/6] Génération de données synthétiques (Mouvement Brownien Géométrique)...
    Nombre de prix : 2000

[2/6] Calcul des log-rendements...
    Nombre de rendements : 1999 | Moyenne : 0.000473 | Écart-type : 0.020012

[3/6] Extraction des caractéristiques Fourier (fenêtre=64, top_k=5)...
    Forme des caractéristiques Fourier : (1999, 10)

[4/6] Extraction des caractéristiques ondelettes (wavelet=db4, niveau=3)...
    Forme des caractéristiques ondelettes : (1999, 12)

[5/6] Construction des séquences (seq_length=60, batch_size=32)...
    Forme de X : (1939, 60, 23) | Forme de y : (1939, 1) | Nombre de caractéristiques : 23

[6/6] Entraînement du modèle StockLSTM (input=23, hidden=128, layers=2, epochs=50)...
    Paramètres entraînables : 198,017

Époque    1/50 | Perte entraînement : 0.985234 | Perte validation : 0.982145 | LR : 1.00e-03
Époque   10/50 | Perte entraînement : 0.921456 | Perte validation : 0.918723 | LR : 1.00e-03
...

============================================================
RÉSULTATS FINAUX
============================================================
  MSE entraînement (dernière époque) : 0.912345
  MSE validation   (dernière époque) : 0.915678
  MSE validation   (meilleure)       : 0.910234
============================================================
```

---

## Caractéristiques du modèle

| Composant | Détails |
|-----------|---------|
| **Signal de base** | Log-rendements `r_t = ln(P_t / P_{t-1})` |
| **Features Fourier** | Top-K fréquences + magnitudes sur fenêtres glissantes (2K features) |
| **Features Ondelettes** | (level+1) × 3 statistiques (énergie, MAV, écart-type) |
| **Séquence d'entrée** | Fenêtre de `seq_length` pas de temps |
| **Modèle** | LSTM multi-couches + tête FC (Linear → ReLU → Dropout → Linear) |
| **Cible** | Rendement au pas de temps t+1 |
| **Perte** | MSE (Mean Squared Error) |
| **Optimiseur** | Adam avec scheduler ReduceLROnPlateau |

---

## Remarques techniques

- **Pas de fuite de données** : la standardisation est ajustée uniquement sur les données d'entraînement.
- **Division chronologique** : pas de mélange aléatoire des séries temporelles.
- **Gradient clipping** : `max_norm=1.0` pour la stabilité de l'entraînement LSTM.
- **Early stopping** : arrêt automatique si la perte de validation ne s'améliore pas.
- **Reproductibilité** : graine aléatoire fixée pour les données synthétiques.
