"""
Module de calcul des rendements logarithmiques financiers.

Les log-rendements r_t = ln(P_t / P_{t-1}) présentent l'avantage d'être
approximativement stationnaires, ce qui est une propriété souhaitable pour
l'entraînement des modèles de séries temporelles.
"""

import numpy as np


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Calcule les log-rendements à partir d'une série de prix.

    La transformation logarithmique r_t = ln(P_t / P_{t-1}) rend la série
    approximativement stationnaire et normalise les variations de prix.

    Paramètres
    ----------
    prices : np.ndarray
        Série temporelle de prix, de forme (n,) ou (n, 1).
        Les valeurs doivent être strictement positives.

    Retourne
    --------
    np.ndarray
        Tableau des log-rendements de longueur n-1. Les NaN ou valeurs
        invalides sont remplacés par zéro.

    Lève
    ----
    ValueError
        Si le tableau d'entrée contient moins de 2 éléments.

    Exemples
    --------
    >>> import numpy as np
    >>> prix = np.array([100.0, 102.0, 101.0, 103.0])
    >>> compute_log_returns(prix)
    array([ 0.01980263, -0.00985222,  0.01960894])
    """
    prices = np.asarray(prices, dtype=np.float64).flatten()

    if len(prices) < 2:
        raise ValueError(
            "Le tableau de prix doit contenir au moins 2 éléments pour "
            "calculer les rendements."
        )

    # Remplacer les prix nuls ou négatifs par NaN pour éviter log(0) ou log(-x)
    prix_valides = np.where(prices > 0, prices, np.nan)

    # Calcul des log-rendements : r_t = ln(P_t / P_{t-1})
    with np.errstate(invalid="ignore", divide="ignore"):
        rendements = np.log(prix_valides[1:] / prix_valides[:-1])

    # Remplacer les NaN et Inf éventuels par zéro
    rendements = np.where(np.isfinite(rendements), rendements, 0.0)

    return rendements
