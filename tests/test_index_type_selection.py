from vector_search.faiss_manager import select_index_type


def test_small_collection_uses_flatip():
    assert select_index_type(500) == "FlatIP"


def test_large_collection_uses_ivf():
    assert select_index_type(50000) == "IVF"


def test_boundary_at_threshold_uses_flatip():
    assert select_index_type(10000) == "FlatIP"


def test_just_above_threshold_uses_ivf():
    assert select_index_type(10001) == "IVF"


def test_custom_threshold():
    assert select_index_type(100, threshold=50) == "IVF"
    assert select_index_type(30, threshold=50) == "FlatIP"
