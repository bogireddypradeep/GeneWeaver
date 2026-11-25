
"""
GeneWeaver - FASTA Parser

Reads DNA sequences from FASTA files using Biopython.
"""

from pathlib import Path
from typing import Dict

from Bio import SeqIO


VALID_BASES = set("ACGTN")


def parse_fasta(file_path: str) -> Dict[str, str]:
    """
    Parse a FASTA file and return its sequences.

    Args:
        file_path: Path to the FASTA file.

    Returns:
        Dictionary mapping sequence identifiers to DNA sequences.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains no valid sequences or invalid bases.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    sequences = {}

    try:
        for record in SeqIO.parse(str(path), "fasta"):
            sequence_id = record.id
            sequence = str(record.seq).upper()

            if not sequence:
                raise ValueError(
                    f"Sequence '{sequence_id}' is empty."
                )

            invalid_bases = set(sequence) - VALID_BASES

            if invalid_bases:
                raise ValueError(
                    f"Sequence '{sequence_id}' contains invalid bases: "
                    f"{sorted(invalid_bases)}"
                )

            if sequence_id in sequences:
                raise ValueError(
                    f"Duplicate sequence identifier: {sequence_id}"
                )

            sequences[sequence_id] = sequence

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            f"Unable to parse FASTA file: {exc}"
        ) from exc

    if not sequences:
        raise ValueError("No sequences found in FASTA file.")

    return sequences
