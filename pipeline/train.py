"""
Module d'entraînement du modèle LSTM.

Ce module implémente la boucle d'entraînement standard avec :
  - Fonction de perte MSE (Mean Squared Error).
  - Optimiseur Adam.
  - Scheduler de taux d'apprentissage ReduceLROnPlateau.
  - Arrêt anticipé (early stopping) basé sur la perte de validation.
  - Visualisation des courbes de perte (train/val).
"""

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Moteur non-interactif pour les environnements sans affichage
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 15,
    log_interval: int = 10,
) -> Dict[str, List[float]]:
    """Entraîne le modèle LSTM avec validation et arrêt anticipé.

    Boucle d'entraînement complète incluant :
      - Phase d'entraînement : propagation avant, calcul de la perte,
        rétropropagation, mise à jour des poids.
      - Phase de validation : calcul de la perte sans gradient.
      - Scheduler LR : réduit le taux d'apprentissage si la perte de
        validation ne s'améliore pas.
      - Early stopping : arrêt si la perte de validation ne s'améliore
        pas pendant `patience` époques consécutives.

    Paramètres
    ----------
    model : nn.Module
        Le modèle LSTM à entraîner.
    train_loader : DataLoader
        DataLoader pour l'ensemble d'entraînement.
    val_loader : DataLoader
        DataLoader pour l'ensemble de validation.
    epochs : int, optionnel
        Nombre maximum d'époques d'entraînement. Par défaut : 100.
    lr : float, optionnel
        Taux d'apprentissage initial pour Adam. Par défaut : 1e-3.
    device : str, optionnel
        Dispositif de calcul ('cpu' ou 'cuda'). Par défaut : 'cpu'.
    patience : int, optionnel
        Nombre d'époques sans amélioration avant l'arrêt anticipé.
        Par défaut : 15.
    log_interval : int, optionnel
        Fréquence d'affichage des pertes (en nombre d'époques).
        Par défaut : 10.

    Retourne
    --------
    dict
        Dictionnaire contenant l'historique des pertes :
          - 'train_loss' : liste des pertes d'entraînement par époque.
          - 'val_loss' : liste des pertes de validation par époque.
    """
    model = model.to(device)
    critere = nn.MSELoss()
    optimiseur = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiseur, mode="min", factor=0.5, patience=patience // 3
    )

    historique: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    meilleure_perte_val = float("inf")
    compteur_patience = 0
    meilleurs_poids = None

    for epoque in range(1, epochs + 1):
        # --- Phase d'entraînement ---
        model.train()
        perte_train_totale = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimiseur.zero_grad()
            predictions = model(X_batch)
            perte = critere(predictions, y_batch)
            perte.backward()
            # Gradient clipping pour la stabilité de l'entraînement LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiseur.step()
            perte_train_totale += perte.item() * len(X_batch)

        perte_train_moy = perte_train_totale / len(train_loader.dataset)

        # --- Phase de validation ---
        model.eval()
        perte_val_totale = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                perte = critere(predictions, y_batch)
                perte_val_totale += perte.item() * len(X_batch)

        perte_val_moy = perte_val_totale / len(val_loader.dataset)

        # Mise à jour du scheduler
        scheduler.step(perte_val_moy)

        # Enregistrement de l'historique
        historique["train_loss"].append(perte_train_moy)
        historique["val_loss"].append(perte_val_moy)

        # Affichage périodique
        if epoque % log_interval == 0 or epoque == 1:
            lr_actuel = optimiseur.param_groups[0]["lr"]
            print(
                f"Époque {epoque:4d}/{epochs} | "
                f"Perte entraînement : {perte_train_moy:.6f} | "
                f"Perte validation : {perte_val_moy:.6f} | "
                f"LR : {lr_actuel:.2e}"
            )

        # --- Early stopping ---
        if perte_val_moy < meilleure_perte_val:
            meilleure_perte_val = perte_val_moy
            compteur_patience = 0
            # Sauvegarde des meilleurs poids
            meilleurs_poids = {
                k: v.clone() for k, v in model.state_dict().items()
            }
        else:
            compteur_patience += 1
            if compteur_patience >= patience:
                print(
                    f"\nArrêt anticipé à l'époque {epoque} "
                    f"(aucune amélioration depuis {patience} époques)."
                )
                break

    # Restauration des meilleurs poids
    if meilleurs_poids is not None:
        model.load_state_dict(meilleurs_poids)
        print(f"Meilleurs poids restaurés (perte val = {meilleure_perte_val:.6f}).")

    return historique


def plot_losses(history: Dict[str, List[float]], save_path: str = None) -> None:
    """Trace les courbes de perte d'entraînement et de validation.

    Paramètres
    ----------
    history : dict
        Dictionnaire retourné par `train_model` contenant les clés
        'train_loss' et 'val_loss'.
    save_path : str, optionnel
        Chemin de sauvegarde de la figure. Si None, la figure est affichée.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs_range, history["train_loss"], label="Perte entraînement", linewidth=2)
    ax.plot(epochs_range, history["val_loss"], label="Perte validation", linewidth=2)

    ax.set_xlabel("Époque")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Courbes de perte — Entraînement vs Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Courbe de perte sauvegardée : {save_path}")
    else:
        plt.show()

    plt.close(fig)
