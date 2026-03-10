"""
Script d'orchestration principal du pipeline de prévision de rendements financiers.

Ce script exécute le pipeline complet de bout en bout :
  1. Génération de données synthétiques (Mouvement Brownien Géométrique) ou
     chargement d'un fichier CSV.
  2. Calcul des log-rendements stationnaires.
  3. Extraction des caractéristiques Fourier (spectre dynamique).
  4. Extraction des caractéristiques ondelettes (DWT multi-niveaux).
  5. Construction des séquences et des DataLoaders.
  6. Instanciation et entraînement du modèle StockLSTM.
  7. Affichage des résultats et traçage des courbes de perte.

Utilisation
-----------
    python main.py
    python main.py --csv_path donnees.csv --epochs 50 --seq_length 30
    python main.py --epochs 100 --batch_size 64 --hidden_size 256 --lr 0.0005
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from data.returns import compute_log_returns
from features.fourier import dynamic_fourier_features
from features.wavelets import wavelet_features
from models.lstm import StockLSTM
from pipeline.dataset import build_sequences, create_dataloaders
from pipeline.train import plot_losses, train_model


def generer_prix_synthetiques(
    n: int = 2000,
    mu: float = 0.0005,
    sigma: float = 0.02,
    prix_initial: float = 100.0,
    seed: int = 42,
) -> np.ndarray:
    """Génère une série de prix par Mouvement Brownien Géométrique (MBG).

    Le MBG est un modèle standard en finance pour simuler l'évolution des
    prix d'actifs : dP = P * (μ dt + σ dW) où dW est un processus de Wiener.

    Paramètres
    ----------
    n : int, optionnel
        Nombre de pas de temps à générer. Par défaut : 2000.
    mu : float, optionnel
        Dérive (rendement moyen par pas). Par défaut : 0.0005.
    sigma : float, optionnel
        Volatilité (écart-type des rendements). Par défaut : 0.02.
    prix_initial : float, optionnel
        Prix initial de la série. Par défaut : 100.0.
    seed : int, optionnel
        Graine aléatoire pour la reproductibilité. Par défaut : 42.

    Retourne
    --------
    np.ndarray
        Série de prix de longueur n.
    """
    rng = np.random.default_rng(seed)
    # Incréments browniens
    chocs = rng.normal(loc=0.0, scale=1.0, size=n)
    # Log-rendements simulés
    log_rendements = (mu - 0.5 * sigma**2) + sigma * chocs
    # Reconstitution des prix par exponentiation cumulative
    log_prix = np.log(prix_initial) + np.cumsum(log_rendements)
    return np.exp(log_prix)


def charger_prix_csv(chemin: str, colonne_prix: str = "close") -> np.ndarray:
    """Charge une série de prix depuis un fichier CSV.

    Paramètres
    ----------
    chemin : str
        Chemin vers le fichier CSV.
    colonne_prix : str, optionnel
        Nom de la colonne contenant les prix. Par défaut : 'close'.

    Retourne
    --------
    np.ndarray
        Série de prix sous forme de tableau NumPy.
    """
    df = pd.read_csv(chemin)
    colonnes_lower = {c.lower(): c for c in df.columns}

    # Recherche insensible à la casse
    if colonne_prix.lower() in colonnes_lower:
        col = colonnes_lower[colonne_prix.lower()]
    elif len(df.columns) == 1:
        col = df.columns[0]
    else:
        # Tenter les colonnes numériques typiques
        candidates = ["close", "price", "adj close", "adjusted_close"]
        col = None
        for c in candidates:
            if c in colonnes_lower:
                col = colonnes_lower[c]
                break
        if col is None:
            raise ValueError(
                f"Colonne de prix introuvable dans {chemin}. "
                f"Colonnes disponibles : {list(df.columns)}"
            )

    prix = pd.to_numeric(df[col], errors="coerce").dropna().values
    return prix.astype(np.float64)


def parse_arguments() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline de prévision de rendements financiers "
            "avec LSTM + Fourier + Ondelettes"
        )
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Chemin vers un fichier CSV de prix (optionnel). "
             "Si absent, des données synthétiques sont générées.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre maximum d'époques d'entraînement. Par défaut : 50.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=60,
        help="Longueur des séquences d'entrée (fenêtre temporelle). Par défaut : 60.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Taille des mini-lots. Par défaut : 32.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Nombre d'unités cachées LSTM. Par défaut : 128.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Taux d'apprentissage initial. Par défaut : 0.001.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Nombre de couches LSTM empilées. Par défaut : 2.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Probabilité de dropout. Par défaut : 0.2.",
    )
    parser.add_argument(
        "--fourier_window",
        type=int,
        default=64,
        help="Taille de la fenêtre Fourier glissante. Par défaut : 64.",
    )
    parser.add_argument(
        "--fourier_top_k",
        type=int,
        default=5,
        help="Nombre de fréquences Fourier dominantes à extraire. Par défaut : 5.",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="Famille d'ondelettes PyWavelets. Par défaut : 'db4'.",
    )
    parser.add_argument(
        "--wavelet_level",
        type=int,
        default=3,
        help="Nombre de niveaux de décomposition ondelettes. Par défaut : 3.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Patience pour l'early stopping. Par défaut : 15.",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Chemin de sauvegarde de la courbe de perte (ex: pertes.png).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositif de calcul ('cpu' ou 'cuda'). Détection automatique si absent.",
    )
    return parser.parse_args()


def main() -> None:
    """Fonction principale d'orchestration du pipeline."""
    args = parse_arguments()

    # Détection automatique du dispositif
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Dispositif utilisé : {device}")

    # =========================================================================
    # Étape 1 : Chargement ou génération des données de prix
    # =========================================================================
    if args.csv_path is not None:
        print(f"\n[1/6] Chargement des prix depuis : {args.csv_path}")
        try:
            prix = charger_prix_csv(args.csv_path)
        except Exception as e:
            print(f"Erreur lors du chargement du CSV : {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\n[1/6] Génération de données synthétiques (Mouvement Brownien Géométrique)...")
        prix = generer_prix_synthetiques(n=2000)

    print(f"    Nombre de prix : {len(prix)}")

    # =========================================================================
    # Étape 2 : Calcul des log-rendements
    # =========================================================================
    print("\n[2/6] Calcul des log-rendements...")
    rendements = compute_log_returns(prix)
    print(
        f"    Nombre de rendements : {len(rendements)} | "
        f"Moyenne : {rendements.mean():.6f} | "
        f"Écart-type : {rendements.std():.6f}"
    )

    # =========================================================================
    # Étape 3 : Extraction des caractéristiques Fourier
    # =========================================================================
    print(
        f"\n[3/6] Extraction des caractéristiques Fourier "
        f"(fenêtre={args.fourier_window}, top_k={args.fourier_top_k})..."
    )
    fourier_feats = dynamic_fourier_features(
        rendements,
        window_size=args.fourier_window,
        num_top_frequencies=args.fourier_top_k,
    )
    print(f"    Forme des caractéristiques Fourier : {fourier_feats.shape}")

    # =========================================================================
    # Étape 4 : Extraction des caractéristiques ondelettes
    # =========================================================================
    print(
        f"\n[4/6] Extraction des caractéristiques ondelettes "
        f"(wavelet={args.wavelet}, niveau={args.wavelet_level})..."
    )
    wavelet_feats = wavelet_features(
        rendements,
        wavelet=args.wavelet,
        level=args.wavelet_level,
    )
    print(f"    Forme des caractéristiques ondelettes : {wavelet_feats.shape}")

    # =========================================================================
    # Étape 5 : Construction des séquences et DataLoaders
    # =========================================================================
    print(
        f"\n[5/6] Construction des séquences "
        f"(seq_length={args.seq_length}, batch_size={args.batch_size})..."
    )
    X, y, scaler = build_sequences(
        rendements,
        fourier_feats,
        wavelet_feats,
        sequence_length=args.seq_length,
    )
    print(
        f"    Forme de X : {tuple(X.shape)} | "
        f"Forme de y : {tuple(y.shape)} | "
        f"Nombre de caractéristiques : {X.shape[2]}"
    )

    train_loader, val_loader = create_dataloaders(
        X, y, train_ratio=0.8, batch_size=args.batch_size
    )
    print(
        f"    Lots d'entraînement : {len(train_loader)} | "
        f"Lots de validation : {len(val_loader)}"
    )

    # =========================================================================
    # Étape 6 : Entraînement du modèle
    # =========================================================================
    input_size = X.shape[2]
    print(
        f"\n[6/6] Entraînement du modèle StockLSTM "
        f"(input={input_size}, hidden={args.hidden_size}, "
        f"layers={args.num_layers}, epochs={args.epochs})..."
    )

    modele = StockLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    # Affichage du nombre de paramètres
    n_params = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    print(f"    Paramètres entraînables : {n_params:,}")

    historique = train_model(
        modele,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        patience=args.patience,
    )

    # =========================================================================
    # Résultats finaux
    # =========================================================================
    perte_train_finale = historique["train_loss"][-1]
    perte_val_finale = historique["val_loss"][-1]
    perte_val_min = min(historique["val_loss"])

    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"  MSE entraînement (dernière époque) : {perte_train_finale:.6f}")
    print(f"  MSE validation   (dernière époque) : {perte_val_finale:.6f}")
    print(f"  MSE validation   (meilleure)       : {perte_val_min:.6f}")
    print("=" * 60)

    # Traçage des courbes de perte
    plot_losses(historique, save_path=args.save_plot)


if __name__ == "__main__":
    main()
