"""
GeneWeaver - FASTA Parser

Reads and validates DNA sequences from FASTA files using Biopython.
Supports all 15 standard IUPAC DNA nucleotide symbols.
"""

from pathlib import Path
from typing import Dict

from Bio import SeqIO

from src.sequence_validation import validate_dna_sequence


def parse_fasta(file_path: str) -> Dict[str, str]:
    """
    Parse a FASTA file and return its validated DNA sequences.

    Args:
        file_path: Path to the FASTA file.

    Returns:
        Dictionary mapping sequence identifiers to DNA sequences.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains no valid sequences,
                    duplicate IDs, or invalid DNA symbols.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    sequences = {}

    try:
        for record in SeqIO.parse(str(path), "fasta"):
            sequence_id = record.id
            sequence = validate_dna_sequence(
                str(record.seq),
                f"Sequence '{sequence_id}'"
            )

            if sequence_id in sequences:
                raise ValueError(
                    f"Duplicate sequence identifier: {sequence_id}"
                )

            sequences[sequence_id] = sequence

    except (ValueError, TypeError):
        raise

    except Exception as exc:
        raise ValueError(
            f"Unable to parse FASTA file: {exc}"
        ) from exc

    if not sequences:
        raise ValueError("No sequences found in FASTA file.")

    return sequences