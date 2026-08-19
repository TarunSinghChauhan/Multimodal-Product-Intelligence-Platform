from unittest.mock import patch

import pytest

from vector_search.faiss_manager import VectorSearchSystem


def test_build_index_raises_for_unsupported_type():
    system = VectorSearchSystem(embedding_dim=512)
    with pytest.raises(ValueError):
        system.build_index(None, index_type="not_a_real_type")


def test_hybrid_search_combines_weighted_scores():
    system = VectorSearchSystem(embedding_dim=512)

    text_results = [{"product_id": "p1", "score": 0.9}, {"product_id": "p2", "score": 0.5}]
    img_results = [{"product_id": "p1", "score": 0.8}, {"product_id": "p3", "score": 0.6}]

    def fake_search(query_embedding, k=10, index_name="fusion"):
        if index_name == "text":
            return text_results, 1.0
        elif index_name == "image":
            return img_results, 1.0
        return [], 1.0

    with patch.object(system, "search", side_effect=fake_search):
        results = system.hybrid_search(text_emb=None, img_emb=None, alpha=0.6, beta=0.4, k=10)

    p1 = next(r for r in results if r["product_id"] == "p1")
    expected_p1_score = 0.6 * 0.9 + 0.4 * 0.8
    assert p1["score"] == pytest.approx(expected_p1_score)


def test_hybrid_search_respects_k_limit():
    system = VectorSearchSystem(embedding_dim=512)

    text_results = [{"product_id": f"p{i}", "score": 1.0 - i * 0.1} for i in range(10)]

    def fake_search(query_embedding, k=10, index_name="fusion"):
        return text_results, 1.0

    with patch.object(system, "search", side_effect=fake_search):
        results = system.hybrid_search(text_emb=None, img_emb=None, k=3)

    assert len(results) == 3


def test_hybrid_search_sorts_descending_by_combined_score():
    system = VectorSearchSystem(embedding_dim=512)

    text_results = [{"product_id": "low", "score": 0.1}, {"product_id": "high", "score": 0.9}]

    def fake_search(query_embedding, k=10, index_name="fusion"):
        return text_results, 1.0

    with patch.object(system, "search", side_effect=fake_search):
        results = system.hybrid_search(text_emb=None, img_emb=None, k=2)

    assert results[0]["product_id"] == "high"
    assert results[1]["product_id"] == "low"


def test_hybrid_search_product_only_in_one_source_uses_partial_score():
    system = VectorSearchSystem(embedding_dim=512)

    text_results = [{"product_id": "text_only", "score": 1.0}]
    img_results = []

    def fake_search(query_embedding, k=10, index_name="fusion"):
        if index_name == "text":
            return text_results, 1.0
        return img_results, 1.0

    with patch.object(system, "search", side_effect=fake_search):
        results = system.hybrid_search(text_emb=None, img_emb=None, alpha=0.6, beta=0.4, k=10)

    text_only = next(r for r in results if r["product_id"] == "text_only")
    assert text_only["score"] == pytest.approx(0.6)
