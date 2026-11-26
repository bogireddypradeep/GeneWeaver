
import pytest

from src.fasta_parser import parse_fasta


def test_parse_valid_fasta(tmp_path):
    fasta = tmp_path / "valid.fasta"
    fasta.write_text(
        ">gene_1\nACGTACGT\n>gene_2\nTTGGCCAA\n",
        encoding="utf-8",
    )

    sequences = parse_fasta(str(fasta))

    assert len(sequences) == 2
    assert sequences["gene_1"] == "ACGTACGT"
    assert sequences["gene_2"] == "TTGGCCAA"


def test_parse_lowercase_sequences(tmp_path):
    fasta = tmp_path / "lowercase.fasta"
    fasta.write_text(">gene_1\nacgt\n", encoding="utf-8")

    sequences = parse_fasta(str(fasta))

    assert sequences["gene_1"] == "ACGT"


def test_parse_ambiguous_base_n(tmp_path):
    fasta = tmp_path / "ambiguous.fasta"
    fasta.write_text(">gene_1\nACNT\n", encoding="utf-8")

    sequences = parse_fasta(str(fasta))

    assert sequences["gene_1"] == "ACNT"


def test_missing_fasta_file():
    with pytest.raises(FileNotFoundError):
        parse_fasta("data/nonexistent.fasta")


def test_empty_fasta_file(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_fasta(str(fasta))


def test_invalid_dna_bases(tmp_path):
    fasta = tmp_path / "invalid.fasta"
    fasta.write_text(">gene_1\nACGTXYZ\n", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_fasta(str(fasta))


def test_empty_sequence(tmp_path):
    fasta = tmp_path / "empty_sequence.fasta"
    fasta.write_text(">gene_1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_fasta(str(fasta))


def test_duplicate_sequence_identifiers(tmp_path):
    fasta = tmp_path / "duplicate.fasta"
    fasta.write_text(
        ">gene_1\nACGT\n>gene_1\nTTAA\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        parse_fasta(str(fasta))
def test_parse_all_iupac_symbols(tmp_path):
    fasta_file = tmp_path / "iupac.fasta"
    fasta_file.write_text(
        ">iupac_sequence\nACGTRYSWKMBDHVN\n",
        encoding="utf-8"
    )

    sequences = parse_fasta(str(fasta_file))

    assert sequences["iupac_sequence"] == "ACGTRYSWKMBDHVN"


def test_reject_unsupported_dna_symbols(tmp_path):
    fasta_file = tmp_path / "invalid.fasta"
    fasta_file.write_text(
        ">invalid_sequence\nACGTXYZ\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        parse_fasta(str(fasta_file))