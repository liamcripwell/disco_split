import re

from discourse_simp.processing.connectives import *

connectives = [conn for rel in PATTERNS.values() 
                    for sense in rel.values() for conn in sense]

def test_valid_equivalencies():
    for conn, eqs in EQUIVALENCIES.items():
        assert conn in connectives
        for eq in eqs:
            assert eq in connectives

def test_valid_inverses():
    for conn, invs in INVERSES.items():
        assert conn in connectives
        for inv in invs:
            assert inv in connectives
            assert inv != conn

def test_strict_result():
    # as a result
    good = [
        "as a result, he died.",
        "as a result of this, he died."
    ]
    bad = [
        "as a result of a new disease, a man died."
    ]
    pattern = STRICT_PATTERNS["as a result"]
    assert all([re.search(pattern, s) is not None for s in good])
    assert not all([re.search(pattern, s) is not None for s in bad])

def test_strict_in_comparison():
    # in comparison
    good = [
        "in comparison, he died.",
        "in comparison to this, a man died."
    ]
    bad = [
        "in comparison to a man who lived, a man died."
    ]
    pattern = STRICT_PATTERNS["by/in comparison"]
    assert all([re.search(pattern, s) is not None for s in good])
    assert not all([re.search(pattern, s) is not None for s in bad])

def test_strict_besides():
    good = [
        "besides, he died.",
        "besides this, he died."
    ]
    bad = [
        "besides 25 people being injured, one man died."
    ]
    pattern = STRICT_PATTERNS["besides"]
    assert all([re.search(pattern, s) is not None for s in good])
    assert not all([re.search(pattern, s) is not None for s in bad])

def test_strict_rather():
    good = [
        "rather, he died.",
        "rather than this, he died."
    ]
    bad = [
        "rather than living, one man died."
    ]
    pattern = STRICT_PATTERNS["rather"]
    assert all([re.search(pattern, s) is not None for s in good])
    assert not all([re.search(pattern, s) is not None for s in bad])
