"""PAIR inference — embed new items and predict their pairwise correlations.

One-step function: give it raw item text, it embeds the items (same backend/
model the checkpoint was trained on) and scores every pair through the
trained Siamese DNN.

    import pair
    df = pair.predict([
        "I see myself as someone who is talkative.",
        "I see myself as someone who is reserved.",
        "I see myself as someone who is full of energy.",
    ])
    # -> polars DataFrame: item1 | item2 | predicted_r  (3 pairs)

`model` is the same id you trained with -- HF ("Qwen/Qwen3-Embedding-8B"),
Ollama ("qwen3-embedding:8b"), or an already-sanitized model_safe folder
name ("Qwen-Qwen3-Embedding-8B"). It does double duty: sanitized (via the
same "/" "" "-" substitution embed_items.py uses) it's the checkpoint
folder name under models_root, and given in raw form it's also what's used
to embed new items -- no separate embed_model= to keep in sync by hand.
backend= is auto-detected from the id's shape ("/" -> hf, ":" -> ollama)
when not given. No separate bundled/aliased copy: this looks directly in
models_root, which defaults to this repo's own models/ (resolved from
wherever `pair` is installed from -- works with no extra args when
installed editable from within the full repo, same as your training
pipeline). Point models_root= elsewhere if you're using this from a
different checkout.

A backbone is fully self-describing (embedding backend + architecture) once
trained with the current train()/model_training.py: embed_items.py writes
embedding_meta.json straight into models_root/model_safe/ (alongside where
the checkpoint and quantile_transformer_meta.json will land), so for those,
passing `model=` alone is enough -- predict() only ever needs models_root,
nothing from data_root. For checkpoints trained before that meta existed,
or where the checkpoint's model_safe folder doesn't match a plain
substitution of the embedding id (e.g. a hand-renamed folder), pass
`backend=` and `embed_model=` explicitly to override (and optionally
`encoder_dims=`/`head_dims=`/`dropout=`/`use_skip=` if the architecture meta
is also missing).

Input formats:
  - `items`: list[str], a polars/pandas DataFrame with an "item" column (or
    a single column), or a path to a .csv/.tsv/.parquet file with one. All
    unique pairwise combinations are scored.
  - `pairs`: instead of `items`, give explicit pairs to score only those --
    list[tuple[str, str]], a two-column DataFrame, or a .csv/.parquet path.

Two-step alternative (embed once, predict many times against the cache,
without paying the embedding cost twice):

    emb = pair.embed(items, model="text-embedding-3-small", backend="api")
    emb.write_parquet("item_embeddings.parquet")
    ...
    emb = pl.read_parquet("item_embeddings.parquet")
    corrs = pair.predict(pairs=my_pairs, embeddings=emb, model="text-embedding-3-small")

Also runnable from the command line:

    python inference.py --items-file new_scale_items.csv --out predictions.csv
    python inference.py --model intfloat/e5-mistral-7b-instruct --items-file new_scale_items.csv --out predictions.csv

Note: each call reloads the embedding backend from scratch (for HF/Ollama
backends this means reloading the model weights), so it's not free to call
repeatedly in a tight loop. Fine for scoring one new scale at a time; if you
need to score many small batches back to back, batch them into one call
instead of many.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

SCRIPT_DIR = Path(__file__).resolve().parent      # code/
REPO_ROOT = SCRIPT_DIR.parent

sys.path.append(str(SCRIPT_DIR / "modelling"))
sys.path.append(str(SCRIPT_DIR / "data_prep" / "helper_functions"))
from siamese_model import SiameseDNN  # noqa: E402
from embedding_functions import get_embeddings, get_embeddings_API, get_embeddings_HF  # noqa: E402


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #

def _read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, separator="\t" if path.suffix == ".tsv" else ",")


def _load_items(items) -> list:
    if isinstance(items, (str, Path)):
        df = _read_table(items)
        col = "item" if "item" in df.columns else df.columns[0]
        items = df[col].to_list()
    elif hasattr(items, "columns"):  # polars or pandas DataFrame
        col = "item" if "item" in items.columns else list(items.columns)[0]
        items = list(items[col])
    return list(dict.fromkeys(str(x).strip() for x in items if str(x).strip()))


def _load_pairs(pairs) -> list:
    if isinstance(pairs, (str, Path)):
        df = _read_table(pairs)
        cols = df.columns[:2]
        return list(zip(df[cols[0]].to_list(), df[cols[1]].to_list()))
    if hasattr(pairs, "columns"):
        cols = list(pairs.columns)[:2]
        return list(zip(list(pairs[cols[0]]), list(pairs[cols[1]])))
    return [tuple(p) for p in pairs]


def _read_json(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _embeddings_to_dict(embeddings) -> dict:
    """Accept a dict{item: vector}, a DataFrame (item | emb1 | emb2 | ...), or a
    path to one -- as produced by embed() / written straight to parquet -- and
    normalize to dict{item: np.ndarray}."""
    if isinstance(embeddings, dict):
        return {k: np.asarray(v, dtype=np.float32) for k, v in embeddings.items()}
    if isinstance(embeddings, (str, Path)):
        embeddings = _read_table(embeddings)
    cols = list(embeddings.columns)
    item_col = "item" if "item" in cols else cols[0]
    emb_cols = [c for c in cols if c != item_col]
    items = list(embeddings[item_col])
    mat = embeddings.select(emb_cols).to_numpy() if hasattr(embeddings, "select") else embeddings[emb_cols].to_numpy()
    return {t: np.asarray(v, dtype=np.float32) for t, v in zip(items, mat)}


# --------------------------------------------------------------------------- #
# Model / backend resolution -- shared by embed() and predict_correlations()
# --------------------------------------------------------------------------- #

def _resolve_model(model, quantize, backend, embed_model) -> tuple:
    """model_safe from `model` (+ "-Nbit" if quantize), and -- unless already
    given -- embed_model/backend inferred from `model`'s own shape ("/" -> hf,
    ":" -> ollama). One place for this so embed() and predict_correlations()
    can't drift apart on how they resolve the same argument."""
    model = str(model)
    model_safe = model.replace(":", "-").replace("/", "-")
    if quantize and not model_safe.endswith(f"-{quantize}bit"):
        model_safe += f"-{quantize}bit"

    if embed_model is None and ("/" in model or ":" in model):
        embed_model = model
    if backend is None:
        if "/" in model:
            backend = "hf"
        elif ":" in model:
            backend = "ollama"

    return model_safe, backend, embed_model


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #

def _embed_items(item_texts, model_dir, backend=None, embed_model=None, **overrides) -> dict:
    meta = _read_json(Path(model_dir) / "embedding_meta.json")
    backend = backend or meta.get("backend")
    embed_model_id = embed_model or meta.get("model")
    if not backend or not embed_model_id:
        raise ValueError(
            f"No embedding_meta.json found under '{model_dir}' (or it's missing "
            "backend/model). Pass backend= and embed_model= explicitly, e.g. "
            "backend='hf', embed_model='Qwen/Qwen3-Embedding-8B'."
        )

    item_list = pl.DataFrame({"item": item_texts})
    print(f"Embedding {len(item_texts)} unique items via {backend}:{embed_model_id} ...")

    if backend == "hf":
        cfg = {
            "instruction": meta.get("instruction"),
            "batch_size": meta.get("batch_size", 32),
            "max_seq_length": meta.get("max_seq_length", 256),
            "normalize": meta.get("normalize", False),
            "quantize": meta.get("quantize", 0),
        }
        cfg.update(overrides)
        embeddings, _ = get_embeddings_HF(item_list, model=embed_model_id, **cfg)
    elif backend == "ollama":
        cfg = {"include_instruction": meta.get("include_instruction", True)}
        cfg.update(overrides)
        embeddings, _ = get_embeddings(item_list, model=embed_model_id, **cfg)
    elif backend == "api":
        cfg = {
            "provider": meta.get("provider", "openai"),
            "dims": meta.get("dims"),  # None -> get_embeddings_API resolves per-model
            "task_type": meta.get("task_type", "SEMANTIC_SIMILARITY"),
        }
        cfg.update(overrides)
        embeddings, _ = get_embeddings_API(item_list, model=embed_model_id, **cfg)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if isinstance(embeddings, dict):
        return {k: np.asarray(v, dtype=np.float32) for k, v in embeddings.items()}
    return {text: np.asarray(vec, dtype=np.float32) for text, vec in zip(item_texts, embeddings)}


def embed(items, model="Qwen/Qwen3-Embedding-8B", backend=None, embed_model=None,
          quantize=None, models_root=None, **embed_overrides) -> pl.DataFrame:
    """Embed item text -- the first half of predict(), exposed standalone so you
    can cache embeddings to disk and reuse them across multiple predict() calls
    instead of paying the embedding cost twice.

        emb = pair.embed(items, model="text-embedding-3-small", backend="api")
        emb.write_parquet("item_embeddings.parquet")     # cache to disk
        ...
        emb = pl.read_parquet("item_embeddings.parquet")  # later / another run
        corrs = pair.predict(pairs=my_pairs, embeddings=emb, model="text-embedding-3-small")

    `model`/`backend`/`quantize`/`embed_model` resolve exactly as in
    predict() (see its docstring) -- model_safe is used to find
    embedding_meta.json under models_root/model_safe/ if backend/embed_model
    aren't given explicitly.

    Returns a polars DataFrame: item | emb1 | emb2 | ... -- pass it straight
    into predict(embeddings=...), or write_parquet()/write_csv() it to save.
    """
    models_root = Path(models_root) if models_root else REPO_ROOT / "models"
    item_texts = _load_items(items)
    if not item_texts:
        raise ValueError("No items to embed.")

    model_safe, backend, embed_model = _resolve_model(model, quantize, backend, embed_model)
    if quantize is not None:
        embed_overrides["quantize"] = quantize

    emb_by_item = _embed_items(item_texts, models_root / model_safe, backend=backend,
                                embed_model=embed_model, **embed_overrides)

    emb_matrix = np.vstack([emb_by_item[t] for t in item_texts])
    df = pl.from_numpy(emb_matrix, schema=[f"emb{i + 1}" for i in range(emb_matrix.shape[1])], orient="row")
    df.insert_column(0, pl.Series("item", item_texts))
    return df


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def predict_correlations(
    items=None,
    pairs=None,
    embeddings=None,
    model="Qwen/Qwen3-Embedding-8B",
    quantize=None,
    backend=None,
    embed_model=None,
    encoder_dims=None,
    head_dims=None,
    dropout=None,
    use_skip=None,
    models_root=None,
    eval_chunk=8192,
    **embed_overrides,
) -> pl.DataFrame:
    """Embed items and predict pairwise correlations with a trained PAIR checkpoint.

    Provide either `items` (score every pairwise combination) or `pairs`
    (score only the given pairs). See module docstring for input formats.

    `model` does double duty: sanitized ("/" and ":" -> "-") it's the
    model_safe checkpoint folder name looked up under models_root, and given
    in raw HF/Ollama form it's also the id used to embed new items -- so in
    the common case, one argument is enough; you don't need to separately
    pass embed_model= and keep it in sync with model= by hand. backend= is
    likewise auto-detected from the shape of `model` ("/" -> hf, ":" ->
    ollama) when not given explicitly. Pass backend=/embed_model= yourself to
    override either (e.g. a checkpoint whose folder name doesn't match a
    plain substitution of its embedding id).

    `embeddings`: skip the embedding step and reuse precomputed vectors --
    pass the DataFrame embed() returned (or something matching its shape:
    item | emb1 | emb2 | ..., a dict{item: vector}, or a path to a saved
    one). Every item in `items`/`pairs` must have a row in `embeddings`;
    `model` is still needed to pick the checkpoint, but backend=/embed_model=
    /quantize= are irrelevant here since nothing gets re-embedded. Good for
    a two-step flow: embed once with embed(), cache to disk, then call
    predict(embeddings=...) as many times as you like against that cache.

    `quantize`: 0, 4, or 8 -- selects the quantized checkpoint variant.
    model="Qwen/Qwen3-Embedding-8B" + quantize=8 resolves to the
    "Qwen-Qwen3-Embedding-8B-8bit" checkpoint (matching embed_items.py's own
    "-Nbit" naming from train(..., quantize=8)), and is also passed through
    to the embedding step so new items are embedded at the same precision
    the checkpoint was trained on. Leave as None if `model`/model_safe
    already includes the "-Nbit" suffix itself.

    Only needs models_root -- everything predict() reads (checkpoint,
    quantile_transformer_meta.json, embedding_meta.json) lives together
    under models_root/model_safe/. Defaults to this repo's own models/.

    Returns a polars DataFrame with columns item1, item2, predicted_r.
    """
    models_root = Path(models_root) if models_root else REPO_ROOT / "models"

    model_safe, backend, embed_model = _resolve_model(model, quantize, backend, embed_model)
    if quantize is not None:
        embed_overrides["quantize"] = quantize

    model_dir = models_root / model_safe

    if not (model_dir / "dnn_siamese_cor.pt").exists():
        raise FileNotFoundError(
            f"No checkpoint found at {model_dir / 'dnn_siamese_cor.pt'}. "
            f"'{model_safe}' should be a model_safe folder name under models_root "
            f"({models_root})"
            + (f" -- quantize={quantize} appended '-{quantize}bit' to model='{model}'" if quantize else "")
            + " -- train one with modelling/train.py, or pass models_root= "
              "pointing at the repo that has it."
        )

    if pairs is not None:
        pair_list = _load_pairs(pairs)
        item_texts = list(dict.fromkeys(t for pair in pair_list for t in pair))
    elif items is not None:
        item_texts = _load_items(items)
        pair_list = list(itertools.combinations(item_texts, 2))
    else:
        raise ValueError("Provide either `items` (all pairwise combinations) or `pairs` (specific pairs).")

    if not pair_list:
        return pl.DataFrame({"item1": [], "item2": [], "predicted_r": []})

    if embeddings is not None:
        emb_by_item = _embeddings_to_dict(embeddings)
        missing = [t for t in item_texts if t not in emb_by_item]
        if missing:
            raise ValueError(
                f"{len(missing)} item(s) from items/pairs not found in `embeddings` "
                f"(first: {missing[0]!r}). Pass embeddings covering every item, or "
                "omit embeddings= to have predict() embed them now."
            )
    else:
        emb_by_item = _embed_items(item_texts, model_dir, backend=backend,
                                    embed_model=embed_model, **embed_overrides)
    emb_dim = len(next(iter(emb_by_item.values())))

    # ---- architecture, from quantile_transformer_meta.json when available ----
    arch_meta = _read_json(model_dir / "quantile_transformer_meta.json")
    encoder_dims = tuple(encoder_dims) if encoder_dims is not None else tuple(arch_meta.get("encoder_dims", (384,)))
    head_dims = tuple(head_dims) if head_dims is not None else tuple(arch_meta.get("head_dims", (256, 130)))
    dropout = dropout if dropout is not None else arch_meta.get("dropout", 0.0)
    use_skip = use_skip if use_skip is not None else arch_meta.get("use_skip", False)

    scaler_path = model_dir / "quantile_transformer.joblib"
    ckpt_path = model_dir / "dnn_siamese_cor.pt"
    import joblib

    scaler = joblib.load(scaler_path)
    aux_dim = scaler.n_features_in_

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SiameseDNN(
        emb_dim=emb_dim, aux_dim=aux_dim,
        encoder_dims=encoder_dims, head_dims=head_dims,
        dropout=dropout, use_skip=use_skip,
    ).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()

    # ---- features ----
    e1 = np.vstack([emb_by_item[a] for a, b in pair_list])
    e2 = np.vstack([emb_by_item[b] for a, b in pair_list])
    e1n = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2n = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
    global_sim = (e1n * e2n).sum(axis=1).astype(np.float32)  # raw cosine, matches training's global_sim
    aux = scaler.transform(global_sim.reshape(-1, 1)).astype(np.float32)

    e1_t = torch.tensor(e1, dtype=torch.float32, device=device)
    e2_t = torch.tensor(e2, dtype=torch.float32, device=device)
    aux_t = torch.tensor(aux, dtype=torch.float32, device=device)

    preds = []
    with torch.no_grad():
        for s in range(0, len(pair_list), eval_chunk):
            sl = slice(s, s + eval_chunk)
            preds.append(torch.tanh(net(e1_t[sl], e2_t[sl], aux_t[sl])).cpu().numpy())
    preds = np.concatenate(preds)

    return pl.DataFrame({
        "item1": [a for a, b in pair_list],
        "item2": [b for a, b in pair_list],
        "predicted_r": preds,
    })


def to_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot a long item1/item2/predicted_r frame into a symmetric item x item matrix."""
    items = sorted(set(df["item1"].to_list()) | set(df["item2"].to_list()))
    idx = {it: i for i, it in enumerate(items)}
    mat = np.eye(len(items), dtype=np.float32)
    for a, b, r in df.iter_rows():
        i, j = idx[a], idx[b]
        mat[i, j] = mat[j, i] = r
    out = pl.DataFrame(mat, schema=items)
    out.insert_column(0, pl.Series("item", items))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Embed new items and predict their pairwise correlations with a trained PAIR checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-8B",
                    help="HF/Ollama id ('Qwen/Qwen3-Embedding-8B') or an already-sanitized "
                         "model_safe folder name ('Qwen-Qwen3-Embedding-8B'). Doubles as both the "
                         "checkpoint lookup name and the embedding id when given in raw form -- "
                         "backend is auto-detected from '/' vs ':' unless --backend is given.")
    p.add_argument("--quantize", type=int, choices=[0, 4, 8], default=None,
                    help="Select the quantized checkpoint variant, e.g. --quantize 8 with "
                         "--model Qwen/Qwen3-Embedding-8B resolves to the '...-8bit' checkpoint "
                         "and embeds new items at the same precision.")
    p.add_argument("--items-file", default=None,
                    help="CSV/TSV/parquet with an 'item' column (or single column). Scores all pairwise combinations.")
    p.add_argument("--pairs-file", default=None,
                    help="CSV/TSV/parquet with two item columns. If given, scores only these pairs (overrides --items-file).")
    p.add_argument("--backend", choices=["hf", "ollama", "api"], default=None,
                    help="Override the auto-detected/embedding_meta.json backend.")
    p.add_argument("--embed-model", default=None,
                    help="Override the embedding model id (default: same as --model, or read from "
                         "embedding_meta.json for a sanitized --model).")
    p.add_argument("--models-root", default=str(REPO_ROOT / "models"),
                    help="Root of trained checkpoints -- everything predict() reads lives under "
                         "models_root/model_safe/.")
    p.add_argument("--out", default="predictions.csv", help="Output path (.csv or .parquet).")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if not args.items_file and not args.pairs_file:
        raise SystemExit("Provide --items-file or --pairs-file.")

    df = predict_correlations(
        items=args.items_file if not args.pairs_file else None,
        pairs=args.pairs_file,
        model=args.model,
        quantize=args.quantize,
        backend=args.backend,
        embed_model=args.embed_model,
        models_root=args.models_root,
    )

    out_path = Path(args.out)
    if out_path.suffix == ".parquet":
        df.write_parquet(out_path)
    else:
        df.write_csv(out_path)
    print(f"\nWrote {df.height:,} predicted pairs -> {out_path}")


if __name__ == "__main__":
    main()
