from llm_pipelines.description_generation import DescriptionGenerator


def make_generator():
    return DescriptionGenerator(api_key="fake-key")


def test_quality_score_full_keyword_overlap():
    gen = make_generator()
    description = "blue cotton shirt casual"
    attributes = "blue cotton shirt casual"
    score = gen.calculate_quality_score(description, attributes)
    # overlap should be 1.0, so score = (readability/100)*0.4 + 1.0*0.6
    assert score >= 0.6


def test_quality_score_zero_keyword_overlap():
    gen = make_generator()
    description = "completely unrelated wording here"
    attributes = "blue cotton shirt casual"
    score = gen.calculate_quality_score(description, attributes)
    # overlap = 0, so score is purely (readability/100)*0.4
    assert score <= 0.4


def test_quality_score_is_higher_with_more_overlap():
    gen = make_generator()
    attributes = "blue cotton shirt casual comfortable"

    low_overlap_desc = "a nice product for everyone"
    high_overlap_desc = "blue cotton shirt casual comfortable design"

    low_score = gen.calculate_quality_score(low_overlap_desc, attributes)
    high_score = gen.calculate_quality_score(high_overlap_desc, attributes)

    assert high_score > low_score


def test_quality_score_handles_empty_attributes_without_crashing():
    gen = make_generator()
    score = gen.calculate_quality_score("some description text", "")
    assert isinstance(score, float)


def test_quality_score_returns_float_in_reasonable_range():
    gen = make_generator()
    score = gen.calculate_quality_score(
        "A comfortable blue cotton shirt perfect for everyday wear.",
        "blue cotton shirt comfortable"
    )
    assert isinstance(score, float)
    assert -1.0 <= score <= 2.0
