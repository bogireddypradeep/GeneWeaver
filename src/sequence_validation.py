"""
GeneWeaver - DNA Sequence Validation

Validates DNA sequences using the 15 standard IUPAC nucleotide symbols.
"""

from typing import Set

# All 15 standard IUPAC DNA nucleotide symbols.
VALID_DNA_BASES: Set[str] = set("ACGTRYSWKMBDHVN")


def validate_dna_sequence(
    sequence: str,
    sequence_name: str = "DNA sequence"
) -> str:
    """
    Validate and normalize a DNA sequence.

    Returns:
        Uppercase DNA sequence.

    Raises:
        TypeError: If sequence is not a string.
        ValueError: If sequence is empty or contains unsupported symbols.
    """

    if not isinstance(sequence, str):
        raise TypeError(f"{sequence_name} must be a string.")

    normalized = sequence.upper()

    if not normalized:
        raise ValueError(f"{sequence_name} must not be empty.")

    invalid_bases = set(normalized) - VALID_DNA_BASES

    if invalid_bases:
        raise ValueError(
            f"{sequence_name} contains invalid DNA symbols: "
            f"{sorted(invalid_bases)}"
        )

    return normalized