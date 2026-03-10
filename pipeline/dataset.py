"""
Module de construction du jeu de données pour l'entraînement du modèle LSTM.

Ce module fournit des fonctions pour :
  1. Concaténer les caractéristiques (rendements, Fourier, ondelettes).
  2. Standardiser chaque colonne (centrer-réduire) avec un StandardScaler.
  3. Construire des fenêtres glissantes pour l'entraînement supervisé.
  4. Créer des DataLoaders PyTorch pour l'entraînement et la validation.
"""

from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def build_sequences(
    returns: np.ndarray,
    fourier_feats: np.ndarray,
    wavelet_feats: np.ndarray,
    sequence_length: int = 60,
    scaler: StandardScaler = None,
) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    """Construit les séquences d'entrée et les cibles pour le modèle LSTM.

    Cette fonction :
      1. Concatène les rendements, les caractéristiques Fourier et ondelettes
         colonne par colonne en un seul tableau de caractéristiques.
      2. Standardise chaque colonne (moyenne=0, écart-type=1) via StandardScaler.
      3. Crée des fenêtres glissantes de longueur `sequence_length` comme
         séquences d'entrée.
      4. Définit la cible y[i] = rendement au pas de temps i + sequence_length
         (prédiction du rendement à t+1).

    Paramètres
    ----------
    returns : np.ndarray
        Série des rendements de forme (n,).
    fourier_feats : np.ndarray
        Caractéristiques Fourier de forme (n, n_fourier).
    wavelet_feats : np.ndarray
        Caractéristiques ondelettes de forme (n, n_wavelet).
    sequence_length : int, optionnel
        Longueur des fenêtres glissantes (pas de temps par séquence).
        Par défaut : 60.
    scaler : StandardScaler, optionnel
        Scaler pré-ajusté à utiliser pour la transformation (utile pour
        l'ensemble de validation/test). Si None, un nouveau scaler est ajusté.

    Retourne
    --------
    X_tensor : torch.Tensor
        Tenseur d'entrée de forme (num_samples, sequence_length, num_features).
    y_tensor : torch.Tensor
        Tenseur cible de forme (num_samples, 1).
    scaler : StandardScaler
        Le scaler ajusté (utile pour transformer les données de test).

    Lève
    ----
    ValueError
        Si les tableaux de caractéristiques n'ont pas la même longueur.
    """
    returns = np.asarray(returns, dtype=np.float64).flatten()
    n = len(returns)

    # Vérification de la cohérence des dimensions
    if fourier_feats.shape[0] != n or wavelet_feats.shape[0] != n:
        raise ValueError(
            f"Toutes les caractéristiques doivent avoir la même longueur. "
            f"Reçu : returns={n}, fourier={fourier_feats.shape[0]}, "
            f"wavelet={wavelet_feats.shape[0]}"
        )

    # Concaténation des caractéristiques : [rendements | Fourier | ondelettes]
    returns_col = returns.reshape(-1, 1)
    features_concat = np.hstack([returns_col, fourier_feats, wavelet_feats])

    # Standardisation (centrer-réduire) chaque colonne
    if scaler is None:
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features_concat)
    else:
        features_norm = scaler.transform(features_concat)

    # Construction des fenêtres glissantes
    num_samples = n - sequence_length
    if num_samples <= 0:
        raise ValueError(
            f"Pas assez de données pour créer des séquences. "
            f"n={n}, sequence_length={sequence_length}"
        )

    num_features = features_norm.shape[1]
    X = np.zeros((num_samples, sequence_length, num_features), dtype=np.float32)
    y = np.zeros((num_samples, 1), dtype=np.float32)

    for i in range(num_samples):
        # Séquence d'entrée : pas de temps [i, i + sequence_length)
        X[i] = features_norm[i : i + sequence_length]
        # Cible : rendement au pas de temps i + sequence_length (t+1)
        y[i, 0] = features_norm[i + sequence_length, 0]  # indice 0 = rendements

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    return X_tensor, y_tensor, scaler


def create_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.8,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Crée les DataLoaders d'entraînement et de validation.

    La division est chronologique (pas de mélange aléatoire) pour respecter
    la structure temporelle des données financières.

    Paramètres
    ----------
    X : torch.Tensor
        Tenseur d'entrée de forme (num_samples, sequence_length, num_features).
    y : torch.Tensor
        Tenseur cible de forme (num_samples, 1).
    train_ratio : float, optionnel
        Proportion des données utilisées pour l'entraînement. Par défaut : 0.8.
    batch_size : int, optionnel
        Taille des mini-lots pour l'entraînement. Par défaut : 32.

    Retourne
    --------
    train_loader : DataLoader
        DataLoader pour l'ensemble d'entraînement (avec mélange activé).
    val_loader : DataLoader
        DataLoader pour l'ensemble de validation (sans mélange).
    """
    n = len(X)
    n_train = int(n * train_ratio)

    # Division chronologique : les données les plus récentes constituent la validation
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    dataset_train = TensorDataset(X_train, y_train)
    dataset_val = TensorDataset(X_val, y_val)

    # Shuffle=True pour l'entraînement, False pour la validation (ordre temporel)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
