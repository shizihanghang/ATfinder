"""Validate standard amino-acid sequences used by ATfinder and its ESM cache."""
from __future__ import annotations


CANONICAL_ORDER = tuple("ARNDCQEGHILKMFPSTWYV")
PHYS_ENCODING_POLICY = "canonical20_strict_v1"
# Keep this identifier unchanged: the existing 1953x640 ESM cache is valid.
ESM_INPUT_POLICY = "canonical20_direct_v1"
_ALLOWED_SYMBOLS = frozenset(CANONICAL_ORDER)


def validate_sequences(sequences) -> None:
    for row, sequence in enumerate(sequences):
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"sequence row {row}: expected a nonempty residue string")
        for position, symbol in enumerate(sequence, start=1):
            if symbol not in _ALLOWED_SYMBOLS:
                raise ValueError(
                    f"sequence row {row}: unsupported residue {symbol!r} "
                    f"at position {position}; only the 20 canonical amino acids are accepted"
                )


def esm_sequences(sequences) -> list[str]:
    """Validate ESM input strings without computing or changing features."""
    sequences = list(sequences)
    validate_sequences(sequences)
    return sequences
