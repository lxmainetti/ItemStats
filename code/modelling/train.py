"""PAIR training — embed a backbone's items, train a checkpoint, validate it.

Repo-only dev tool, not part of the pip-installable `pair` package's public
surface (that package is predict()-only, no training data attached). This
requires a full clone of the repo, since embed_items.py / model_training.py /
model_validation.py need data/raw/*.parquet (the training data pipeline)
that isn't distributed via pip. Lives in modelling/ (not the code/ package
root) precisely so it doesn't get swept into the pip package's wheel.

One function wrapping embed_items.py -> model_training.py -> model_validation.py
as subprocesses (kept as separate processes deliberately, so each heavy stage
-- loading a multi-GB embedding model, then a torch training loop -- gets its
own clean GPU memory rather than accumulating across stages in one process).

    from train import train
    model_safe = train("Qwen/Qwen3-Embedding-8B")

Loop over several backbones the same way:

    for hf_id in ["Qwen/Qwen3-Embedding-8B", "intfloat/e5-mistral-7b-instruct"]:
        train(hf_id)

Use the resulting model_safe with pair.predict(items, model=model_safe) once
trained (pip install -e . from the repo root first if you haven't).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent   # code/modelling
CODE_DIR = SCRIPT_DIR.parent                    # code/
PYTHON = sys.executable

EMBED_SCRIPT = CODE_DIR / "data_prep" / "embed_items.py"
TRAIN_SCRIPT = SCRIPT_DIR / "model_training.py"
VALIDATE_SCRIPT = SCRIPT_DIR / "model_validation.py"


def _sh(cmd):
    """Run cmd, stream output live, return it as one string. Raises on nonzero exit."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[1]} exited with code {proc.returncode}")
    return "".join(lines)


def train(model, backend="hf", quantize=0, skip_embed=False, validate=True, no_plots=True, **kwargs):
    """Embed -> train -> (optionally) validate a PAIR checkpoint for one embedding backbone.

    model:      HF id ("Qwen/Qwen3-Embedding-8B"), Ollama tag, or API model name to
                embed items with. This is the *embedding* model, not a bundled predict()
                alias -- training always starts from a real backend id.
    backend:    "hf" | "ollama" | "api"
    quantize:   0 (full precision), 4, or 8 -- load the embedding backbone in reduced
                precision (hf backend only, via bitsandbytes). embed_items.py appends
                a "-Nbit" suffix to model_safe when this is set, so e.g. quantize=8
                on "Qwen/Qwen3-Embedding-8B" trains and saves under
                "Qwen-Qwen3-Embedding-8B-8bit". Use the same quantize= value with
                pair.predict(..., quantize=8) to load the matching checkpoint.
    skip_embed: reuse embeddings already on disk for this backbone instead of re-embedding.
    validate:   also run holdout+validation scoring and append a row to
                performance_logbook.xlsx (set False to skip for speed).
    no_plots:   skip saving scatter/training-curve PNGs (default True; sweeps rarely need them).
    kwargs:     passed straight through as extra CLI flags to embed_items.py
                (e.g. batch_size=16, dims=1024).

    Returns model_safe, the folder name under data/raw/ and models/ the trained
    checkpoint was saved under -- pass this straight to pair.predict(model=...).
    """
    plot_flag = ["--no-plots"] if no_plots else []
    if quantize:
        kwargs["quantize"] = quantize
    extra = [f"--{k.replace('_', '-')}" for k, v in kwargs.items() if v is True]
    extra += [x for k, v in kwargs.items() if v is not True for x in (f"--{k.replace('_', '-')}", str(v))]

    if skip_embed:
        model_safe = model.replace(":", "-").replace("/", "-")
        if quantize:
            model_safe += f"-{quantize}bit"
    else:
        out = _sh([PYTHON, str(EMBED_SCRIPT), "--backend", backend, "--model", model,
                   "--splits", "train", "holdout", "validation", *extra])
        model_safe = re.findall(r"model_safe\s*=\s*(\S+)", out)[-1]

    _sh([PYTHON, str(TRAIN_SCRIPT), "--emb-model", model_safe, *plot_flag])
    if validate:
        _sh([PYTHON, str(VALIDATE_SCRIPT), "--emb-model", model_safe, *plot_flag])

    print(f"\nDone: {model_safe}")
    return model_safe
