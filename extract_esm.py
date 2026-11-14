"""Extract genuine frozen ESM-2 150M layer-30 mean-residue features.

Run once after downloading the final dataset. If --weights is not given,
facebookresearch/esm downloads the official checkpoint into the Torch cache.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from ours.config import atfinder_config
from ours.data import load_multilabel_data, load_protocol
from ours.residues import ESM_INPUT_POLICY, esm_sequences


MODEL_NAME = "esm2_t30_150M_UR50D"


def sequence_sha256(sequences: list[str]) -> str:
    return hashlib.sha256("\n".join(sequences).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ATfinder ESM-2 input cache")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Optional official esm2_t30_150M_UR50D.pt checkpoint")
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate dataset/splits without downloading weights")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    if args.batch_size < 1:
        parser.error("--batch-size must be >=1")
    if args.weights is not None and not args.weights.is_file():
        parser.error(f"Official weights not found: {args.weights}")

    cfg = atfinder_config(data_dir=args.data_dir, device=args.device)
    sequences, targets, _ = load_multilabel_data(cfg)
    load_protocol(cfg, len(sequences))
    if len(sequences) != 1953 or targets.shape != (1953, 12):
        raise ValueError("This release expects the final 1953×12 dataset")
    if int(targets.sum()) != 4794:
        raise ValueError("Unexpected observed-positive label count")
    model_inputs = esm_sequences(sequences)
    original_digest = sequence_sha256(sequences)
    esm_digest = sequence_sha256(model_inputs)
    print(f"Dataset validated: {len(sequences)} peptides, {int(targets.sum())} positives")
    print(f"Sequence SHA256: {original_digest}")
    if args.check_only:
        return

    try:
        import esm
    except ImportError as exc:
        raise SystemExit("Install the official ESM package with `pip install fair-esm`.") from exc

    if args.weights is None:
        print("Downloading/loading official ESM-2 150M weights...", flush=True)
        try:
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        except Exception as exc:
            raise SystemExit(
                "Unable to download ESM-2 weights. On a machine with internet access "
                "download the official esm2_t30_150M_UR50D.pt and rerun with --weights."
            ) from exc
    else:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(args.weights))
    if getattr(model, "num_layers", None) != 30 or getattr(model, "embed_dim", None) != 640:
        raise ValueError("Checkpoint must be ESM-2 150M (30 layers, 640-dimensional output)")
    model = model.eval().to(args.device)
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=None)
    embeddings = torch.empty((len(sequences), 640), dtype=torch.float32)

    with torch.inference_mode():
        for begin in range(0, len(sequences), args.batch_size):
            batch = model_inputs[begin:begin + args.batch_size]
            _, _, tokens = batch_converter([(str(begin + j), seq) for j, seq in enumerate(batch)])
            output = model(tokens.to(args.device), repr_layers=[30], return_contacts=False)
            hidden = output["representations"][30]
            for j, seq in enumerate(batch):
                # Exclude BOS/EOS and padding. No sequence truncation.
                embeddings[begin + j] = hidden[j, 1:len(seq)+1].mean(dim=0).float().cpu()
            print(f"ESM: {min(begin+len(batch),len(sequences))}/{len(sequences)}", flush=True)

    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("ESM embeddings contain NaN/Inf")

    cache = {
        "embeddings": embeddings,
        "model_name": MODEL_NAME,
        "representation_layer": 30,
        "pooling": "mean_residues_excluding_bos_eos",
        "truncation": None,
        "sequence_sha256": original_digest,
        "esm_sequence_sha256": esm_digest,
        "esm_input_policy": ESM_INPUT_POLICY,
        "sequence_lengths": [len(s) for s in sequences],
        "global_idx": list(range(len(sequences))),
    }
    path = args.data_dir / "esm2_t30_150m_mean.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pt.tmp")
    torch.save(cache, temp)
    temp.replace(path)
    print(f"Saved verified ESM features: {path} {tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()
