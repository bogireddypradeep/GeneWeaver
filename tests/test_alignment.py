
import pytest

from src.alignment import smith_waterman, get_substitution_score


# --------------------------------------------------
# 1. Basic alignment tests
# --------------------------------------------------

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
    assert result.query_start == 0
    assert result.query_end == 5
    assert result.reference_start == 2
    assert result.reference_end == 7


def test_mismatch():
    result = smith_waterman("AAAA", "TTTT")

    assert result.score == 0
    assert result.aligned_query == ""
    assert result.aligned_reference == ""


# --------------------------------------------------
# 2. Input validation tests
# --------------------------------------------------

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


def test_non_string_query():
    with pytest.raises(TypeError):
        smith_waterman(None, "ACGT")


def test_non_string_reference():
    with pytest.raises(TypeError):
        smith_waterman("ACGT", None)


# --------------------------------------------------
# 3. Case sensitivity and IUPAC tests
# --------------------------------------------------

def test_case_insensitive_alignment():
    result = smith_waterman("acgt", "ACGT")

    assert result.score == 8
    assert result.aligned_query == "ACGT"
    assert result.aligned_reference == "ACGT"


def test_ambiguous_base_n():
    result = smith_waterman("ACNT", "ACNT")

    assert result.score == 6
    assert result.aligned_query == "ACNT"
    assert result.aligned_reference == "ACNT"


def test_all_iupac_symbols_supported():
    result = smith_waterman(
        "ACGTRYSWKMBDHVN",
        "ACGTRYSWKMBDHVN"
    )

    assert result.score == 8


def test_ambiguous_symbols_receive_zero_score():
    assert get_substitution_score("N", "N", 2, -1) == 0
    assert get_substitution_score("R", "Y", 2, -1) == 0


def test_canonical_match_score():
    assert get_substitution_score("A", "A", 2, -1) == 2


def test_canonical_mismatch_score():
    assert get_substitution_score("A", "T", 2, -1) == -1


def test_only_ambiguous_sequences_have_zero_score():
    result = smith_waterman("NNNN", "NNNN")

    assert result.score == 0
    assert result.aligned_query == ""
    assert result.aligned_reference == ""


# --------------------------------------------------
# 4. Single-base and boundary tests
# --------------------------------------------------

def test_single_base_match():
    result = smith_waterman("A", "A")

    assert result.score == 2
    assert result.aligned_query == "A"
    assert result.aligned_reference == "A"
    assert result.query_start == 0
    assert result.query_end == 1
    assert result.reference_start == 0
    assert result.reference_end == 1


def test_single_base_mismatch():
    result = smith_waterman("A", "T")

    assert result.score == 0
    assert result.aligned_query == ""
    assert result.aligned_reference == ""


def test_match_at_end_of_sequences():
    result = smith_waterman("TTACGT", "GGACGT")

    assert result.score == 8
    assert result.aligned_query == "ACGT"
    assert result.aligned_reference == "ACGT"
    assert result.query_start == 2
    assert result.query_end == 6
    assert result.reference_start == 2
    assert result.reference_end == 6


def test_match_at_beginning_of_sequences():
    result = smith_waterman("ACGTGG", "ACGTCC")

    assert result.score == 8
    assert result.aligned_query == "ACGT"
    assert result.aligned_reference == "ACGT"
    assert result.query_start == 0
    assert result.query_end == 4
    assert result.reference_start == 0
    assert result.reference_end == 4


# --------------------------------------------------
# 5. Gap handling tests
# --------------------------------------------------

def test_alignment_with_gap():
    result = smith_waterman(
        "ACGT",
        "AGT",
        gap_penalty=-1
    )

    assert result.score == 5
    assert result.aligned_query == "ACGT"
    assert result.aligned_reference == "A-GT"
    assert result.query_start == 0
    assert result.query_end == 4
    assert result.reference_start == 0
    assert result.reference_end == 3


# --------------------------------------------------
# 6. Different sequence lengths
# --------------------------------------------------

@pytest.mark.parametrize(
    "sequence_length",
    [1, 10, 50, 100, 250]
)
def test_different_sequence_lengths(sequence_length):
    sequence = "ACGT" * (sequence_length // 4)
    sequence += "ACGT"[:sequence_length % 4]

    result = smith_waterman(sequence, sequence)

    assert result.score == sequence_length * 2
    assert result.aligned_query == sequence
    assert result.aligned_reference == sequence


# --------------------------------------------------
# 7. Custom scoring tests
# --------------------------------------------------

def test_custom_match_score():
    result = smith_waterman(
        "ACGT",
        "ACGT",
        match_score=3
    )

    assert result.score == 12


def test_custom_mismatch_score():
    assert get_substitution_score("A", "T", 2, -3) == -3
