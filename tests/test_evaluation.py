from disco_split.evaluation.utils import is_accurate


def test_is_accurate():
    assert is_accurate("before that", 95, 90, "earlier") # equivalent connective - True
    assert is_accurate("before that", 95, 5, "later") # inverted connective (inverted order) - True
    assert not is_accurate("before that", 95, 90, "later") # inverted connective (same order) - False
    assert not is_accurate("before that", 95, 5, "previously") # equivalent connective (inverted order) - False
    assert not is_accurate("before that", 95, 90, "consequently") # non-equivalent connective - False