"""Psychometric item embedding — standalone script.

Script version of embed_items.ipynb. Generates semantic embeddings for every
unique item text via one of three backends (local Ollama, hosted API, local
HuggingFace/sentence-transformers) and writes an item-keyed embedding parquet
per data split (training / holdout / validation).

Backend + model default to whatever embed_items.ipynb currently has active
(hosted OpenAI API, text-embedding-3-large, 3072 dims) -- override via CLI to
use a different backend/model without touching the file:

    python embed_items.py                                    # current default (API)
    python embed_items.py --backend hf --model Qwen/Qwen3-Embedding-8B --batch-size 64
    python embed_items.py --backend ollama --model qwen3-embedding:8b
    python embed_items.py --splits train holdout               # skip validation

The notebook (embed_items.ipynb) is untouched and still works standalone;
this script is a drop-in equivalent for batch/HPC use.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent            # code/data_prep
REPO_ROOT = SCRIPT_DIR.parent.parent                      # project root

# Import the embedding-backend functions from the modular helper, resolved
# relative to this script's location so it works regardless of caller CWD.
sys.path.append(str(SCRIPT_DIR / "helper_functions"))
from embedding_functions import get_embeddings, get_embeddings_API, get_embeddings_HF  # noqa: E402

SPLIT_PREFIXES = {"train": "", "holdout": "holdout_", "validation": "validation_"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate item embeddings for one or more PAIR data splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data-root", default=str(REPO_ROOT / "data" / "raw"),
                    help="Root of the raw parquet data (item_list.parquet per split, embeddings output).")
    p.add_argument("--models-root", default=str(REPO_ROOT / "models"),
                    help="Root of trained checkpoints. embedding_meta.json is written under "
                         "models_root/model_safe/ (alongside where the checkpoint will later "
                         "land), not under data_root, so predict() only ever needs models_root.")
    p.add_argument("--splits", nargs="+", default=["train", "holdout", "validation"],
                    choices=list(SPLIT_PREFIXES.keys()),
                    help="Which data splits to embed.")

    p.add_argument("--backend", choices=["api", "ollama", "hf"], default="hf",
                    help="Embedding backend. 'api' = hosted OpenAI/Google, 'ollama' = local Ollama, "
                         "'hf' = local HuggingFace/sentence-transformers.")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-8B",
                    help="Model name/id. Backend-specific, e.g. 'text-embedding-3-large' (api), "
                         "'qwen3-embedding:8b' (ollama), 'Qwen/Qwen3-Embedding-8B' (hf).")

    # ---- api backend ----
    p.add_argument("--provider", choices=["openai", "google"], default="openai",
                    help="[api backend] Hosted embedding provider.")
    p.add_argument("--dims", type=int, default=None,
                    help="[api backend] Output embedding dimensionality. Left unset, resolves "
                         "per-model (1536 for text-embedding-3-small, 3072 for -large, etc.).")
    p.add_argument("--task-type", default="SEMANTIC_SIMILARITY",
                    help="[api backend, google only] Gemini task type.")

    # ---- ollama backend ----
    p.add_argument("--no-instruction", action="store_true", default=False,
                    help="[ollama backend] Disable the task-instruction prefix (only affects qwen3 models).")

    # ---- hf backend ----
    p.add_argument("--instruction", default=None,
                    help="[hf backend] Override the default psychometric instruction prompt.")
    p.add_argument("--batch-size", type=int, default=32,
                    help="[hf backend] Encoding batch size.")
    p.add_argument("--max-seq-length", type=int, default=256,
                    help="[hf backend] Max token sequence length.")
    p.add_argument("--normalize", action="store_true", default=False,
                    help="[hf backend] L2-normalize output embeddings.")
    p.add_argument("--quantize", type=int, choices=[0, 4, 8], default=0,
                    help="[hf backend] Load the model in 4-bit or 8-bit precision (0 = full precision).")

    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    data_root = Path(args.data_root)
    splits = [SPLIT_PREFIXES[s] for s in args.splits]

    
    # ---- Save embedding config so inference.py can re-embed new items the same way ----
    model_safe = args.model.replace(":", "-").replace("/", "-")

    meta = {
        "model_safe": model_safe,
        "backend": args.backend,
        "model": args.model,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.backend == "api":
        meta.update({"provider": args.provider, "dims": args.dims, "task_type": args.task_type})
    elif args.backend == "ollama":
        meta.update({"include_instruction": not args.no_instruction})
    elif args.backend == "hf":
        meta.update({
            "instruction": args.instruction,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "normalize": args.normalize,
            "quantize": args.quantize,
        })
    # Lives under models_root, not data_root -- it's a description of *how to
    # re-embed*, needed at predict() time right alongside the checkpoint,
    # not the (large, training-only) raw embedding parquets in data_root.
    meta_dir = Path(args.models_root) / model_safe
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "embedding_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved embedding meta -> {meta_path}")

    # ---- Embed Items ----
    for split in splits:
        label = split if split else "train"
        print(f"\n=== Split: {label} ===")

        # ---- Read in data ----
        item_list = (
            pl.read_parquet(data_root / f"{split}item_list.parquet")
            .unique(subset="item", keep="first", maintain_order=True)
        )
        print(f"{item_list.height:,} unique items")

        # ---- Generate embeddings ----
        if args.backend == "ollama":
            embeddings, model_safe = get_embeddings(
                item_list, model=args.model, include_instruction=not args.no_instruction,
            )
        elif args.backend == "hf":
            embeddings, model_safe = get_embeddings_HF(
                item_list, model=args.model, instruction=args.instruction,
                batch_size=args.batch_size, max_seq_length=args.max_seq_length,
                normalize=args.normalize, quantize=args.quantize,
            )
        else:  # api
            embeddings, model_safe = get_embeddings_API(
                item_list, model=args.model, provider=args.provider,
                dims=args.dims, task_type=args.task_type,
            )

        # ---- Save data ----
        emb_matrix = np.vstack(list(embeddings.values()) if isinstance(embeddings, dict) else embeddings)
        item_embeddings_df = pl.from_numpy(
            emb_matrix,
            schema=[f"emb{i + 1}" for i in range(emb_matrix.shape[1])],
            orient="row",
        )
        item_embeddings_df.insert_column(0, item_list.get_column("item"))

        out_dir = data_root / model_safe
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}embeddings_raw.parquet"
        item_embeddings_df.write_parquet(out_path)
        print(f"Wrote {item_embeddings_df.shape} -> {out_path}")

    

    print(f"\nDone. model_safe = {model_safe}")
    return model_safe


if __name__ == "__main__":
    main()
