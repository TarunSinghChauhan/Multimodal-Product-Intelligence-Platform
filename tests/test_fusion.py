import numpy as np
import pytest

from embeddings.utils.fusion import fuse_embeddings


def test_fused_output_is_l2_normalized():
    image_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    text_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = fuse_embeddings(image_emb, text_emb)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0)


def test_fusion_respects_weights():
    image_emb = np.array([[1.0, 0.0]])
    text_emb = np.array([[0.0, 1.0]])
    result = fuse_embeddings(image_emb, text_emb, image_weight=0.4, text_weight=0.6)
    expected_unnormalized = np.array([0.4, 0.6])
    expected = expected_unnormalized / np.linalg.norm(expected_unnormalized)
    assert np.allclose(result[0], expected)


def test_fusion_raises_on_shape_mismatch():
    image_emb = np.zeros((3, 512))
    text_emb = np.zeros((3, 256))
    with pytest.raises(ValueError):
        fuse_embeddings(image_emb, text_emb)


def test_fusion_handles_all_zero_row_without_nan():
    image_emb = np.array([[0.0, 0.0]])
    text_emb = np.array([[0.0, 0.0]])
    result = fuse_embeddings(image_emb, text_emb)
    assert not np.any(np.isnan(result))


def test_fusion_preserves_batch_size():
    image_emb = np.random.rand(10, 512)
    text_emb = np.random.rand(10, 512)
    result = fuse_embeddings(image_emb, text_emb)
    assert result.shape == (10, 512)
