"""
Module de calcul des caractéristiques par Transformée en Ondelettes Discrète (DWT).

Ce module utilise la bibliothèque PyWavelets (pywt) pour décomposer la série
des rendements en composantes basse fréquence (approximation) et haute fréquence
(détails) à plusieurs niveaux.

  - Coefficients d'approximation (cA) : capturent la tendance lisse et le
    comportement basse fréquence du marché.
  - Coefficients de détail (cD1, cD2, ...) : capturent les composantes haute
    fréquence, les sauts soudains et les chocs de marché à différentes échelles.

Pour chaque niveau, des statistiques récapitulatives sont calculées sur une
fenêtre glissante afin de produire des caractéristiques de taille fixe par
pas de temps.
"""

import numpy as np
import pywt
from scipy.interpolate import interp1d


def _statistiques_fenetre(
    coeffs: np.ndarray, longueur_originale: int
) -> np.ndarray:
    """Calcule les statistiques (énergie, MAV, écart-type) et rééchantillonne.

    Pour un tableau de coefficients d'ondelettes (plus court que le signal
    original du fait du sous-échantillonnage de la DWT), cette fonction
    calcule trois statistiques scalaires et les rééchantillonne à la
    longueur du signal original par interpolation linéaire.

    Paramètres
    ----------
    coeffs : np.ndarray
        Tableau 1D de coefficients d'ondelettes.
    longueur_originale : int
        Longueur du signal original (longueur cible après rééchantillonnage).

    Retourne
    --------
    np.ndarray
        Tableau de forme (longueur_originale, 3) contenant, pour chaque
        pas de temps rééchantillonné :
          - Colonne 0 : énergie locale (somme des carrés sur fenêtre glissante)
          - Colonne 1 : valeur absolue moyenne (MAV)
          - Colonne 2 : écart-type local
    """
    n_coeffs = len(coeffs)

    if n_coeffs == 0:
        return np.zeros((longueur_originale, 3))

    if n_coeffs == 1:
        # Cas dégénéré : un seul coefficient, on réplique
        energie = np.array([coeffs[0] ** 2])
        mav = np.array([np.abs(coeffs[0])])
        std = np.array([0.0])
    else:
        # Énergie locale : carré du coefficient
        energie = coeffs ** 2
        # Valeur absolue moyenne
        mav = np.abs(coeffs)
        # Écart-type local (utilise un voisinage de ±1)
        std = np.array(
            [
                np.std(coeffs[max(0, k - 1) : min(n_coeffs, k + 2)])
                for k in range(n_coeffs)
            ]
        )

    # Axe original des coefficients (uniformément répartis sur [0, longueur_originale])
    x_coeffs = np.linspace(0, longueur_originale - 1, n_coeffs)
    x_cible = np.arange(longueur_originale, dtype=float)

    def _interpoler(valeurs: np.ndarray) -> np.ndarray:
        if n_coeffs == 1:
            return np.full(longueur_originale, valeurs[0])
        f = interp1d(x_coeffs, valeurs, kind="linear", fill_value="extrapolate")
        return f(x_cible)

    energie_interp = _interpoler(energie)
    mav_interp = _interpoler(mav)
    std_interp = _interpoler(std)

    return np.column_stack([energie_interp, mav_interp, std_interp])


def wavelet_features(
    returns: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> np.ndarray:
    """Calcule les caractéristiques par Transformée en Ondelettes Discrète (DWT).

    Applique une décomposition multi-niveau en ondelettes discrètes sur la
    série des rendements. Pour chaque niveau de décomposition, trois
    statistiques sont extraites (énergie, MAV, écart-type) et rééchantillonnées
    à la longueur originale du signal.

    La décomposition produit :
      - 1 niveau d'approximation (cA) → basse fréquence / tendance
      - `level` niveaux de détail (cD1, ..., cDn) → haute fréquence / chocs

    Au total : (level + 1) * 3 caractéristiques par pas de temps.

    Paramètres
    ----------
    returns : np.ndarray
        Série temporelle des rendements, de forme (n,).
    wavelet : str, optionnel
        Famille d'ondelettes PyWavelets (ex. 'db4', 'haar', 'sym8').
        Par défaut : 'db4'.
    level : int, optionnel
        Nombre de niveaux de décomposition. Par défaut : 3.

    Retourne
    --------
    np.ndarray
        Tableau de forme (n, (level + 1) * 3) contenant les caractéristiques
        d'ondelettes alignées sur l'axe temporel original.

    Exemples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> r = np.random.randn(200)
    >>> feats = wavelet_features(r, wavelet='db4', level=3)
    >>> feats.shape
    (200, 12)
    """
    returns = np.asarray(returns, dtype=np.float64).flatten()
    n = len(returns)

    # Niveau maximal théorique autorisé par PyWavelets
    niveau_max = pywt.dwt_max_level(n, wavelet)
    niveau_effectif = min(level, niveau_max)

    if niveau_effectif < 1:
        # Signal trop court pour la décomposition : retourner des zéros
        return np.zeros((n, (level + 1) * 3))

    # Décomposition multi-niveau en ondelettes discrètes
    # coefficients[0] = cA (approximation)
    # coefficients[1], ..., coefficients[niveau_effectif] = cD (détails, du plus fin au plus grossier)
    coefficients = pywt.wavedec(returns, wavelet=wavelet, level=niveau_effectif)

    # Calcul des statistiques pour chaque niveau et rééchantillonnage
    liste_caracteristiques = []
    for coeffs_niveau in coefficients:
        stats = _statistiques_fenetre(coeffs_niveau, n)
        liste_caracteristiques.append(stats)

    # Compléter avec des zéros si niveau_effectif < level demandé
    niveaux_manquants = (level + 1) - len(liste_caracteristiques)
    for _ in range(niveaux_manquants):
        liste_caracteristiques.append(np.zeros((n, 3)))

    return np.hstack(liste_caracteristiques)
