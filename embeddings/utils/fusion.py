import numpy as np


def fuse_embeddings(image_embeddings: np.ndarray, text_embeddings: np.ndarray, image_weight: float = 0.4, text_weight: float = 0.6) -> np.ndarray:
    """
    Combine image and text embeddings into a single fused, L2-normalized embedding.
    """
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            f"Shape mismatch: image_embeddings {image_embeddings.shape} "
            f"vs text_embeddings {text_embeddings.shape}"
        )

    fused = (image_weight * image_embeddings) + (text_weight * text_embeddings)
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero for all-zero rows
    return fused / norms
