from llm_pipelines.attribute_extraction import calculate_backoff_wait


def test_first_attempt_wait():
    assert calculate_backoff_wait(0) == 2


def test_second_attempt_wait():
    assert calculate_backoff_wait(1) == 3


def test_third_attempt_wait():
    assert calculate_backoff_wait(2) == 5


def test_wait_increases_with_attempt():
    assert calculate_backoff_wait(3) > calculate_backoff_wait(2) > calculate_backoff_wait(1)
