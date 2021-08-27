from discourse_simp.processing.connectives import PATTERNS
from discourse_simp.processing.utils import strip_adverbial


def test_transform_literal():
    pair = ("I went to lunch.", "Later I returned.")
    connective_pattern = PATTERNS["temporal"]["precedence"]["later"]
    stripped_sent = "I returned."

    assert strip_adverbial(connective_pattern, pair[1]) == stripped_sent


def test_transform_optional():
    pair = ("I went to lunch.", "Afterward, I returned.")
    connective_pattern = PATTERNS["temporal"]["precedence"]["afterward(s)"]
    stripped_sent = "I returned."

    assert strip_adverbial(connective_pattern, pair[1]) == stripped_sent


def test_transform_alternate():
    pair = ("She is very strong.", "In comparison, I am weak.")
    connective_pattern = PATTERNS["comparison"]["contrast"]["by/in comparison"]
    stripped_sent = "I am weak."

    assert strip_adverbial(connective_pattern, pair[1]) == stripped_sent


def test_transform_with_comma():
    pair = ("I am not very strong.", "Also, I am weak.")
    connective_pattern = PATTERNS["expansion"]["conjunction"]["also"]
    stripped_sent = "I am weak."

    assert strip_adverbial(connective_pattern, pair[1]) == stripped_sent


def test_transform_contrast_to():
    pair = ("I am not very strong.", "In contrast to that my dad is.")
    connective_pattern = PATTERNS["comparison"]["contrast"]["by/in contrast"]
    stripped_sent = "to that my dad is."

    assert strip_adverbial(connective_pattern, pair[1]) == stripped_sent
