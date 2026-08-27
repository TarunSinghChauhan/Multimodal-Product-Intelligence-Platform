import numpy as np


def calculate_accuracy(predictions, true_labels) -> float:
    """Fraction of predictions matching true_labels. Both must be equal-length sequences."""
    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)
    if len(predictions) != len(true_labels):
        raise ValueError(
            f"Length mismatch: predictions has {len(predictions)}, true_labels has {len(true_labels)}"
        )
    if len(predictions) == 0:
        return 0.0
    return float((predictions == true_labels).mean())
