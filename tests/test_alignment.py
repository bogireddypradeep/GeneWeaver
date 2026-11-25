
import pytest

from src.alignment import smith_waterman


def test_exact_match():
    result = smith_waterman("ACGT", "ACGT")

    assert result.score == 8
    assert result.aligned_query == "ACGT"
    assert result.aligned_reference == "ACGT"


def test_partial_local_match():
    result = smith_waterman("ACGTACGT", "TTACGTAA")

    assert result.score == 10
    assert result.aligned_query == "ACGTA"
    assert result.aligned_reference == "ACGTA"
    assert result.reference_start == 2
    assert result.reference_end == 7


def test_mismatch():
    result = smith_waterman("AAAA", "TTTT")

    assert result.score == 0
    assert result.aligned_query == ""
    assert result.aligned_reference == ""


def test_empty_query():
    with pytest.raises(ValueError):
        smith_waterman("", "ACGT")


def test_empty_reference():
    with pytest.raises(ValueError):
        smith_waterman("ACGT", "")


def test_invalid_query_characters():
    with pytest.raises(ValueError):
        smith_waterman("ACGTXYZ", "ACGT")


def test_invalid_reference_characters():
    with pytest.raises(ValueError):
        smith_waterman("ACGT", "ACGTXYZ")


def test_case_insensitive_alignment():
    result = smith_waterman("acgt", "ACGT")

    assert result.score == 8
    assert result.aligned_query == "ACGT"
    assert result.aligned_reference == "ACGT"


def test_ambiguous_base_n():
    result = smith_waterman("ACNT", "ACNT")

    assert result.score == 8
    assert result.aligned_query == "ACNT"
    assert result.aligned_reference == "ACNT"
