"""
Module définissant l'architecture du modèle LSTM pour la prévision de rendements.

Le modèle StockLSTM combine :
  - Des couches LSTM multi-niveaux pour capturer les dépendances temporelles
    longues dans les séries financières.
  - Une tête de régression fully-connected pour prédire le rendement à t+1.
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """Réseau de neurones LSTM pour la prévision de rendements financiers.

    Architecture :
      - Couches LSTM multi-niveaux avec dropout entre les couches intermédiaires.
      - Tête fully-connected : Linear(hidden_size, 64) → ReLU → Dropout → Linear(64, 1).
      - Sortie : rendement prédit au pas de temps t+1 (scalaire).

    Paramètres
    ----------
    input_size : int
        Nombre de caractéristiques d'entrée par pas de temps (rendements +
        caractéristiques Fourier + caractéristiques ondelettes).
    hidden_size : int, optionnel
        Nombre d'unités cachées dans chaque couche LSTM. Par défaut : 128.
    num_layers : int, optionnel
        Nombre de couches LSTM empilées. Par défaut : 2.
    dropout : float, optionnel
        Probabilité de dropout entre les couches LSTM (et dans la tête FC).
        Par défaut : 0.2.

    Exemples
    --------
    >>> import torch
    >>> modele = StockLSTM(input_size=11, hidden_size=64, num_layers=2, dropout=0.1)
    >>> x = torch.randn(32, 60, 11)  # (batch, seq_len, features)
    >>> sortie = modele(x)
    >>> sortie.shape
    torch.Size([32, 1])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Couches LSTM empilées (batch_first=True pour (batch, seq, features))
        # Le dropout est appliqué entre les couches intermédiaires
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Tête de régression fully-connected
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagation avant du modèle.

        Paramètres
        ----------
        x : torch.Tensor
            Tenseur d'entrée de forme (batch_size, sequence_length, input_size).

        Retourne
        --------
        torch.Tensor
            Tenseur de forme (batch_size, 1) contenant le rendement prédit
            pour chaque séquence du batch.
        """
        # Propagation à travers les couches LSTM
        # lstm_out : (batch_size, sequence_length, hidden_size)
        # (h_n, c_n) : états cachés finaux
        lstm_out, _ = self.lstm(x)

        # On utilise uniquement le dernier état caché de la séquence
        # dernier_etat : (batch_size, hidden_size)
        dernier_etat = lstm_out[:, -1, :]

        # Prédiction via la tête fully-connected
        sortie = self.fc(dernier_etat)  # (batch_size, 1)

        return sortie
