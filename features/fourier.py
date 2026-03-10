"""
Module de calcul de la Transformée de Fourier Dynamique (DFT) pour les
séries temporelles financières.

Ce module implémente une analyse spectrale sur fenêtres glissantes, en
appliquant :
  1. Un tapering en cloche de cosinus (fenêtre de Tukey) pour réduire les
     fuites spectrales (spectral leakage).
  2. La FFT pour passer dans le domaine fréquentiel.
  3. Un lissage par noyau de Daniell (moyenne mobile uniforme) pour
     stabiliser les estimations spectrales.
  4. L'extraction des N fréquences dominantes comme caractéristiques.
"""

import numpy as np
from scipy.signal import windows
from scipy.ndimage import uniform_filter1d


def _appliquer_tukey(fenetre: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Applique une fenêtre de Tukey (cloche de cosinus) au signal.

    La fenêtre de Tukey est une fenêtre rectangulaire dont les extrémités
    sont progressivement ramenées à zéro via une fonction cosinus relevé.
    Cela réduit les discontinuités aux bords de la fenêtre glissante et
    minimise les fuites spectrales.

    Paramètres
    ----------
    fenetre : np.ndarray
        Signal d'entrée 1D de longueur n.
    alpha : float, optionnel
        Fraction de la fenêtre couverte par les transitions cosinus.
        alpha=0 → fenêtre rectangulaire, alpha=1 → fenêtre de Hann.
        Par défaut : 0.5.

    Retourne
    --------
    np.ndarray
        Signal après application du tapering, de même longueur que l'entrée.
    """
    taper = windows.tukey(len(fenetre), alpha=alpha)
    return fenetre * taper


def _lisser_daniell(spectre: np.ndarray, largeur_noyau: int = 5) -> np.ndarray:
    """Lisse le spectre de puissance par convolution avec un noyau de Daniell.

    Le noyau de Daniell est un noyau rectangulaire uniforme (moyenne mobile).
    Il est équivalent à une convolution avec un filtre passe-bas rectangulaire,
    ce qui réduit la variance des estimations spectrales.

    Paramètres
    ----------
    spectre : np.ndarray
        Densité spectrale de puissance 1D à lisser.
    largeur_noyau : int, optionnel
        Largeur du noyau de Daniell (nombre de points de la moyenne mobile).
        Par défaut : 5.

    Retourne
    --------
    np.ndarray
        Spectre lissé de même longueur que l'entrée.
    """
    if largeur_noyau <= 1:
        return spectre
    return uniform_filter1d(spectre, size=largeur_noyau, mode="reflect")


def dynamic_fourier_features(
    returns: np.ndarray,
    window_size: int = 64,
    step: int = 1,
    num_top_frequencies: int = 5,
    tukey_alpha: float = 0.5,
    kernel_width: int = 5,
) -> np.ndarray:
    """Calcule les caractéristiques spectrales dynamiques sur fenêtres glissantes.

    Pour chaque fenêtre glissante de la série des rendements, cette fonction :
      1. Applique une fenêtre de Tukey (cosine bell taper) pour réduire les
         fuites spectrales.
      2. Calcule la FFT de la fenêtre taperisée.
      3. Calcule la densité spectrale de puissance |FFT|².
      4. Lisse le spectre avec un noyau de Daniell (moyenne mobile uniforme).
      5. Extrait les N fréquences dominantes (magnitudes et fréquences normalisées).

    Le résultat est aligné avec l'axe temporel d'origine : les premières
    `window_size - 1` lignes sont remplies avec des zéros (pas de fenêtre
    complète disponible).

    Paramètres
    ----------
    returns : np.ndarray
        Série temporelle des rendements, de forme (n,).
    window_size : int, optionnel
        Taille de la fenêtre glissante. Par défaut : 64.
    step : int, optionnel
        Pas du déplacement de la fenêtre. Par défaut : 1.
    num_top_frequencies : int, optionnel
        Nombre de fréquences dominantes à extraire. Par défaut : 5.
    tukey_alpha : float, optionnel
        Paramètre alpha de la fenêtre de Tukey. Par défaut : 0.5.
    kernel_width : int, optionnel
        Largeur du noyau de Daniell pour le lissage. Par défaut : 5.

    Retourne
    --------
    np.ndarray
        Tableau de forme (n, num_top_frequencies * 2) contenant, pour chaque
        pas de temps, les magnitudes et les fréquences normalisées des N
        fréquences dominantes, concaténées : [mag_1, ..., mag_N, freq_1, ..., freq_N].
        Les premières `window_size - 1` lignes sont nulles (rembourrage).

    Exemples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> r = np.random.randn(200)
    >>> feats = dynamic_fourier_features(r, window_size=32, num_top_frequencies=3)
    >>> feats.shape
    (200, 6)
    """
    returns = np.asarray(returns, dtype=np.float64).flatten()
    n = len(returns)
    num_features = num_top_frequencies * 2  # magnitudes + fréquences

    # Tableau de sortie initialisé à zéro (aligné avec l'axe temporel)
    resultat = np.zeros((n, num_features), dtype=np.float64)

    # Fréquences normalisées pour une fenêtre de taille window_size
    freqs = np.fft.rfftfreq(window_size)

    for i in range(window_size - 1, n, step):
        # Extraction de la fenêtre glissante
        fenetre = returns[i - window_size + 1 : i + 1]

        # 1. Tapering avec la fenêtre de Tukey
        fenetre_taperisee = _appliquer_tukey(fenetre, alpha=tukey_alpha)

        # 2. FFT et densité spectrale de puissance
        spectre_fft = np.fft.rfft(fenetre_taperisee)
        psd = np.abs(spectre_fft) ** 2

        # 3. Lissage par noyau de Daniell
        psd_lisse = _lisser_daniell(psd, largeur_noyau=kernel_width)

        # 4. Extraction des top-N fréquences dominantes
        # On exclut le composant DC (indice 0) pour éviter la tendance
        psd_sans_dc = psd_lisse.copy()
        psd_sans_dc[0] = 0.0

        indices_top = np.argpartition(psd_sans_dc, -num_top_frequencies)[
            -num_top_frequencies:
        ]
        # Tri par ordre décroissant de puissance
        indices_top = indices_top[np.argsort(psd_sans_dc[indices_top])[::-1]]

        magnitudes = np.sqrt(psd_lisse[indices_top])  # amplitude (racine de PSD)
        freqs_top = freqs[indices_top]

        # 5. Stockage : [mag_1, ..., mag_N, freq_1, ..., freq_N]
        resultat[i, :num_top_frequencies] = magnitudes
        resultat[i, num_top_frequencies:] = freqs_top

    # Remplissage par interpolation pour les pas couverts par step > 1
    if step > 1:
        for j in range(num_features):
            indices_valides = np.arange(window_size - 1, n, step)
            indices_tous = np.arange(window_size - 1, n)
            resultat[window_size - 1 : n, j] = np.interp(
                indices_tous, indices_valides, resultat[indices_valides, j]
            )

    return resultat
