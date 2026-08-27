import pytest

from embeddings.utils.accuracy import calculate_accuracy


def test_perfect_accuracy():
    assert calculate_accuracy([0, 1, 2], [0, 1, 2]) == 1.0


def test_zero_accuracy():
    assert calculate_accuracy([0, 0, 0], [1, 1, 1]) == 0.0


def test_partial_accuracy():
    assert calculate_accuracy([0, 1, 2, 3], [0, 1, 9, 9]) == 0.5


def test_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        calculate_accuracy([0, 1], [0, 1, 2])


def test_empty_inputs_return_zero():
    assert calculate_accuracy([], []) == 0.0
