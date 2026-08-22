from types import SimpleNamespace
from unittest.mock import MagicMock

from llm_pipelines.rag_qa import ProductRAG


def make_doc(text, title, price, rating):
    return SimpleNamespace(
        page_content=text,
        metadata={"title": title, "price": price, "avg_rating": rating},
    )


def make_rag():
    return ProductRAG.__new__(ProductRAG)


def test_retrieve_and_rerank_returns_empty_when_no_docs():
    rag = make_rag()
    rag.vector_store = MagicMock()
    rag.vector_store.similarity_search.return_value = []
    rag.cross_encoder = MagicMock()

    result = rag.retrieve_and_rerank("blue shirt")

    assert result == []
    rag.cross_encoder.predict.assert_not_called()


def test_retrieve_and_rerank_sorts_by_score_descending():
    doc_low = make_doc("low relevance text", "Product A", 10.0, 4.0)
    doc_high = make_doc("high relevance text", "Product B", 20.0, 4.5)
    doc_mid = make_doc("mid relevance text", "Product C", 15.0, 4.2)

    rag = make_rag()
    rag.vector_store = MagicMock()
    rag.vector_store.similarity_search.return_value = [doc_low, doc_high, doc_mid]
    rag.cross_encoder = MagicMock()
    rag.cross_encoder.predict.return_value = [0.2, 0.9, 0.5]

    result = rag.retrieve_and_rerank("query", k=10)

    assert result[0] is doc_high
    assert result[1] is doc_mid
    assert result[2] is doc_low


def test_retrieve_and_rerank_truncates_to_top_three():
    docs = [make_doc(f"text {i}", f"Product {i}", 10.0, 4.0) for i in range(5)]
    scores = [0.1, 0.9, 0.5, 0.7, 0.3]

    rag = make_rag()
    rag.vector_store = MagicMock()
    rag.vector_store.similarity_search.return_value = docs
    rag.cross_encoder = MagicMock()
    rag.cross_encoder.predict.return_value = scores

    result = rag.retrieve_and_rerank("query")

    assert len(result) == 3


def test_generate_answer_builds_context_from_docs_and_calls_llm():
    doc = make_doc("A great blue shirt", "Blue Shirt", 25.0, 4.7)

    rag = make_rag()
    rag.vector_store = MagicMock()
    rag.vector_store.similarity_search.return_value = [doc]
    rag.cross_encoder = MagicMock()
    rag.cross_encoder.predict.return_value = [0.9]

    fake_response = SimpleNamespace(content="It costs $25.")
    rag.llm = MagicMock()
    rag.llm.invoke.return_value = fake_response

    answer = rag.generate_answer("How much is the blue shirt?")

    assert answer == "It costs $25."
    call_args = rag.llm.invoke.call_args[0][0]
    assert "Blue Shirt" in call_args
    assert "25.0" in call_args
    assert "How much is the blue shirt?" in call_args


def test_chat_delegates_to_generate_answer():
    rag = make_rag()
    rag.generate_answer = MagicMock(return_value="an answer")

    result = rag.chat("some query")

    assert result == "an answer"
    rag.generate_answer.assert_called_once_with("some query")
