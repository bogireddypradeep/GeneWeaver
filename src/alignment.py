
"""
GeneWeaver - Optimized CPU Sequence Alignment
Smith-Waterman local alignment with IUPAC DNA support.

Optimizations:
1. Numba JIT compilation for dynamic programming.
2. Rolling score rows to reduce memory consumption.
3. Compact traceback matrix for alignment reconstruction.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from src.sequence_validation import validate_dna_sequence


CANONICAL_BASES = set("ACGT")

# Encode canonical bases as 0-3 and ambiguous bases as 4.
BASE_ENCODING = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


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
    Calculate substitution score.

    Identical canonical bases receive match_score.
    Different canonical bases receive mismatch_score.
    Comparisons involving ambiguous bases receive zero.
    """

    if base1 not in CANONICAL_BASES or base2 not in CANONICAL_BASES:
        return 0

    if base1 == base2:
        return match_score

    return mismatch_score


@njit(cache=True)
def _smith_waterman_kernel(
    query_codes,
    reference_codes,
    match_score,
    mismatch_score,
    gap_penalty,
):
    """
    JIT-compiled Smith-Waterman dynamic programming kernel.

    Uses two rolling score rows and a compact traceback matrix.
    """

    rows = len(query_codes) + 1
    cols = len(reference_codes) + 1

    previous = np.zeros(cols, dtype=np.int32)
    current = np.zeros(cols, dtype=np.int32)

    # 0=stop, 1=diagonal, 2=up, 3=left.
    traceback = np.zeros((rows, cols), dtype=np.uint8)

    best_score = 0
    best_i = 0
    best_j = 0

    for i in range(1, rows):

        current[0] = 0

        for j in range(1, cols):

            q_base = query_codes[i - 1]
            r_base = reference_codes[j - 1]

            if q_base < 4 and r_base < 4:

                if q_base == r_base:
                    substitution = match_score
                else:
                    substitution = mismatch_score

            else:
                substitution = 0

            diagonal = previous[j - 1] + substitution
            up = previous[j] + gap_penalty
            left = current[j - 1] + gap_penalty

            score = max(0, diagonal, up, left)

            current[j] = score

            if score == 0:
                traceback[i, j] = 0
            elif score == diagonal:
                traceback[i, j] = 1
            elif score == up:
                traceback[i, j] = 2
            else:
                traceback[i, j] = 3

            if score > best_score:
                best_score = score
                best_i = i
                best_j = j

        # Swap rows instead of allocating another matrix.
        temp = previous
        previous = current
        current = temp

    return best_score, best_i, best_j, traceback


def smith_waterman(
    query: str,
    reference: str,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2,
) -> AlignmentResult:
    """
    Perform optimized Smith-Waterman local alignment.

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

    # Convert DNA symbols into numeric arrays for Numba.
    query_codes = np.fromiter(
        (BASE_ENCODING.get(base, 4) for base in query),
        dtype=np.int8,
        count=len(query),
    )

    reference_codes = np.fromiter(
        (BASE_ENCODING.get(base, 4) for base in reference),
        dtype=np.int8,
        count=len(reference),
    )

    best_score, best_i, best_j, traceback = _smith_waterman_kernel(
        query_codes,
        reference_codes,
        match_score,
        mismatch_score,
        gap_penalty,
    )

    i = best_i
    j = best_j

    aligned_query = []
    aligned_reference = []

    while i > 0 and j > 0:

        direction = traceback[i, j]

        if direction == 0:
            break

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

    aligned_query.reverse()
    aligned_reference.reverse()

    return AlignmentResult(
        score=int(best_score),
        aligned_query="".join(aligned_query),
        aligned_reference="".join(aligned_reference),
        query_start=i,
        query_end=best_i,
        reference_start=j,
        reference_end=best_j,
    )
