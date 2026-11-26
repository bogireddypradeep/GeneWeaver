
"""
GeneWeaver - CPU Sequence Alignment
Smith-Waterman local alignment with IUPAC DNA support.
"""

from dataclasses import dataclass

import numpy as np

from src.sequence_validation import validate_dna_sequence


CANONICAL_BASES = set("ACGT")


@dataclass
class AlignmentResult:
    """Stores the result of a local sequence alignment."""

    score: int
    aligned_query: str
    aligned_reference: str
    query_start: int
    query_end: int
    reference_start: int
    reference_end: int


def get_substitution_score(
    base1: str,
    base2: str,
    match_score: int,
    mismatch_score: int,
) -> int:
    """
    Calculate the substitution score for two DNA bases.

    Only identical canonical bases receive a positive match score.
    Comparisons involving ambiguous IUPAC symbols receive zero.
    """

    if base1 not in CANONICAL_BASES or base2 not in CANONICAL_BASES:
        return 0

    if base1 == base2:
        return match_score

    return mismatch_score


def smith_waterman(
    query: str,
    reference: str,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2,
) -> AlignmentResult:
    """
    Perform Smith-Waterman local alignment.

    Args:
        query: Query DNA sequence.
        reference: Reference DNA sequence.
        match_score: Score for identical canonical bases.
        mismatch_score: Penalty for different canonical bases.
        gap_penalty: Penalty for insertion or deletion.

    Returns:
        AlignmentResult containing score and aligned sequences.
    """

    query = validate_dna_sequence(query, "Query")
    reference = validate_dna_sequence(reference, "Reference")

    rows = len(query) + 1
    cols = len(reference) + 1

    matrix = np.zeros((rows, cols), dtype=np.int32)

    # 0=stop, 1=diagonal, 2=up, 3=left.
    traceback = np.zeros((rows, cols), dtype=np.uint8)

    best_score = 0
    best_position = (0, 0)

    for i in range(1, rows):
        for j in range(1, cols):

            substitution = get_substitution_score(
                query[i - 1],
                reference[j - 1],
                match_score,
                mismatch_score,
            )

            diagonal = matrix[i - 1, j - 1] + substitution
            up = matrix[i - 1, j] + gap_penalty
            left = matrix[i, j - 1] + gap_penalty

            score = max(0, diagonal, up, left)
            matrix[i, j] = score

            if score == 0:
                traceback[i, j] = 0
            elif score == diagonal:
                traceback[i, j] = 1
            elif score == up:
                traceback[i, j] = 2
            else:
                traceback[i, j] = 3

            if score > best_score:
                best_score = int(score)
                best_position = (i, j)

    i, j = best_position

    aligned_query = []
    aligned_reference = []

    while i > 0 and j > 0 and matrix[i, j] > 0:

        direction = traceback[i, j]

        if direction == 1:
            aligned_query.append(query[i - 1])
            aligned_reference.append(reference[j - 1])
            i -= 1
            j -= 1

        elif direction == 2:
            aligned_query.append(query[i - 1])
            aligned_reference.append("-")
            i -= 1

        elif direction == 3:
            aligned_query.append("-")
            aligned_reference.append(reference[j - 1])
            j -= 1

        else:
            break

    aligned_query.reverse()
    aligned_reference.reverse()

    return AlignmentResult(
        score=best_score,
        aligned_query="".join(aligned_query),
        aligned_reference="".join(aligned_reference),
        query_start=i,
        query_end=best_position[0],
        reference_start=j,
        reference_end=best_position[1],
    )
