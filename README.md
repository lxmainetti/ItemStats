# PAIR — Predicting Associations of Inter-item Relationships

A pipeline (and pip-installable inference package) that predicts the pairwise correlation between any two psychometric items directly from their wording. It's built on text embeddings from a swappable backbone (Qwen3-Embedding-8B by default) and a Siamese DNN trained end-to-end on aggregated open-access survey data.

Given just the *text* of two scale items, the model recovers their empirical inter-item correlation with **Pearson r ≈ 0.87** on out-of-training items from familiar scale families (holdout), and **r ≈ 0.80** on entirely unseen scales it never encountered during training (validation — SurveyBot study, AAID + BFI-10). Noise-ceiling-adjusted (see [Backbone comparison](#-backbone-comparison) below), that's **r ≈ 0.91** holdout / **r ≈ 0.82** validation.

> **Architecture note:** earlier versions fed the model a large auxiliary feature block — transformer cross-encoder scores (NLI, reranker similarity, pair sentiment) and per-item sentiment/emotion classifications, plus an autoencoder-compressed embedding space. Permutation-importance analysis showed none of it added meaningfully over the raw embeddings, so the pipeline was simplified to **raw embeddings + `global_sim`** (cosine similarity between the two items' full embeddings) only. See [Retired stages](#-retired-stages).

## 🚀 Quick start (inference)

The package is `pair` (distribution name `pair-psychometrics`), installed editable from a clone of this repo:

```bash
pip install -e /path/to/PAIR
```

```python
import pair

# one step: embed + score every pairwise combination
corrs = pair.predict([
    "I see myself as someone who is talkative.",
    "I see myself as someone who is reserved.",
    "I see myself as someone who is full of energy.",
])
# -> polars DataFrame: item1 | item2 | predicted_r

# or score specific pairs (e.g. two columns from your own DataFrame)
corrs = pair.predict(pairs=my_df.select(["item_text1", "item_text2"]))

# swap backbone / quantization -- one argument does double duty as both the
# checkpoint lookup name and the embedding id when given in HF/Ollama form
corrs = pair.predict(items, model="intfloat/e5-mistral-7b-instruct")
corrs = pair.predict(items, model="Qwen/Qwen3-Embedding-8B", quantize=8)

# two-step, for reuse: embed once, cache to disk, predict many times without
# re-embedding
emb = pair.embed(items, model="text-embedding-3-small", backend="api")
emb.write_parquet("item_embeddings.parquet")
corrs = pair.predict(pairs=my_pairs, embeddings=emb, model="text-embedding-3-small")
```

`pair.to_matrix(corrs)` pivots the long `item1 | item2 | predicted_r` output into a symmetric item × item matrix. Also runnable from the command line: `python code/inference.py --items-file items.csv --out predictions.csv`.

### ⚠️ Model weights

**Most trained checkpoints are not committed to this repository** — a dozen-plus backbones' worth of `dnn_siamese_cor.pt` files would add several GB to git history, which isn't practical without LFS or external hosting (not yet set up; see [Further plans](#️-further-plans)). A handful of early checkpoints are still tracked from before the multi-backbone sweep, but most of `models/` (everything in the [comparison table](#-backbone-comparison) below) is local-only. `pair.predict()`/`pair.embed()` need `models/<backbone>/` to exist locally (`dnn_siamese_cor.pt`, `quantile_transformer.joblib`, `quantile_transformer_meta.json`) — either train your own with `train()` below, or obtain a copy of the `models/` directory out-of-band.

## 📈 Backbone comparison

All backbones trained on the same 2,843-item training pool, same splits, same architecture (`encoder_dims=(384,)`, `head_dims=(256, 130)`, dropout 0.226). `holdout` = Bainbridge mega-study (item-disjoint, same scale families as training, N=87,153 pairs); `validation` = SurveyBot study (entirely unseen scales, N=33,670 pairs). Noise-aware r divides raw r by a split-half-derived noise ceiling on the human target itself — see `model_validation.py`'s `noise_ceiling()`.

| Backbone | Emb. dim | Holdout r | Holdout r (noise-aware) | Validation r | Validation r (noise-aware) |
|---|---|---|---|---|---|
| intfloat/e5-mistral-7b-instruct | 4096 | 0.884 | 0.923 | 0.763 | 0.786 |
| tencent/KaLM-Embedding-Gemma3-12B-2511 | 3840 | 0.877 | 0.916 | 0.800 | 0.825 |
| **Qwen/Qwen3-Embedding-8B (default)** | 4096 | 0.868 | 0.906 | 0.797 | 0.822 |
| text-embedding-3-large (OpenAI) | 3072 | 0.862 | 0.900 | 0.693 | 0.714 |
| Qwen/Qwen3-Embedding-8B (4-bit) | 4096 | 0.861 | 0.899 | 0.769 | 0.792 |
| Qwen/Qwen3-Embedding-8B (8-bit) | 4096 | 0.853 | 0.890 | 0.749 | 0.771 |
| text-embedding-3-small (OpenAI) | 1536 | 0.837 | 0.874 | 0.667 | 0.687 |
| Qwen/Qwen3-Embedding-4B | 2560 | 0.833 | 0.870 | 0.747 | 0.770 |
| BAAI/bge-m3 | 1024 | 0.815 | 0.851 | 0.587 | 0.604 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 1024 | 0.763 | 0.796 | 0.533 | 0.549 |
| nomic-ai/nomic-embed-text-v2-moe | 768 | 0.757 | 0.790 | 0.546 | 0.562 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.655 | 0.684 | 0.315 | 0.325 |

Qwen3-Embedding-8B ships as the default backbone (local inference, no API dependency, strong validation-set generalization). A few other backbones (`intfloat-e5-large-v2`, `dwulff-mpnet-personality`, `sentence-transformers-all-mpnet-base-v2`, `nvidia-Nemotron-3-Embed-8B-BF16`, plus the 4-bit e5-mistral/Qwen3-4B variants) have trained checkpoints under `models/` but haven't been re-benchmarked against the current pipeline yet.

Full run-by-run history lives in `performance_logbook.xlsx` (one row appended per `model_validation.py` run).

## 🧠 Model

**Siamese DNN** (`code/modelling/siamese_model.py`):

- A shared MLP encoder (`Linear → LayerNorm → GELU → Dropout`, `encoder_dims=(384,)`) applied independently to each item's raw embedding.
- Order-invariant 3-way interaction: `concat[h1 + h2, h1 * h2, |h1 - h2|]`, concatenated with `global_sim` (raw cosine similarity between the two items' full embeddings, quantile-transformed).
- Head MLP (`head_dims=(256, 130)`) → linear output in Fisher-z space; `tanh` applied at inference to map back to Pearson r.
- Trained with Huber loss (β=0.127), AdamW (lr 2e-4, weight decay 0.0023), gradient clipping (1.863), `ReduceLROnPlateau`, and early stopping (patience 20) on an item-disjoint outer-validation split (10% of items).

## 🔄 Pipeline

Each stage reads from disk and writes to disk, so any stage can be re-run in isolation.

1. **Data integration** (`code/data_prep/data_integration.R`) — parses ~32 heterogeneous open-access scales (CSV/TSV/SAV/RDS + codebooks), computes per-scale inter-item correlations (plus split-half correlations for the noise-ceiling calculation), and writes unified training/holdout/validation parquet files to `data/raw/`.
2. **Embedding generation** (`code/data_prep/embed_items.py`, backends in `helper_functions/embedding_functions.py`) — embeds every unique item via a local HF/sentence-transformers model, a local Ollama model, or the OpenAI/Google APIs. Writes an item-keyed embedding parquet per split, plus `embedding_meta.json` under `models/<backbone>/` describing exactly how to re-embed new items later.
3. **Training** (`code/modelling/model_training.py`) — fits the Siamese DNN end-to-end on raw embeddings + `global_sim`, saves the best checkpoint (by outer-val RMSE) plus the fitted `QuantileTransformer` and architecture metadata to `models/<backbone>/`.
4. **Evaluation** (`code/modelling/model_validation.py`) — scores a checkpoint on holdout + validation, computes the split-half noise ceiling and noise-aware r, runs a polarity/sign-concordance diagnostic, and appends a row to `performance_logbook.xlsx`.

**Orchestration** — `train()` in `code/modelling/train.py` wraps steps 2–4 as subprocesses (so each heavy stage gets clean GPU memory):

```python
from train import train
model_safe = train("Qwen/Qwen3-Embedding-8B")                          # embed -> train -> validate
model_safe = train("Qwen/Qwen3-Embedding-8B", quantize=8)              # 8-bit backbone
model_safe = train("intfloat/e5-mistral-7b-instruct", skip_embed=True) # reuse existing embeddings
```

`code/run_pipeline.ipynb` is a thin notebook wrapper around the same function, for looping over several backbones interactively.

## 📊 Data sources

Built on the principles of Open Science. The training pool aggregates **~32 published psychometric scales** from open-access repositories; the full scale-by-scale list with abbreviations and citations lives in [`scale_sources.md`](scale_sources.md). By provenance:

- **`psychTools` R package** (5 scales): Eysenck Personality Inventory (EPI), Big Five Inventory (BFI), Motivational State Questionnaire (MSQ), SAPA Personality Inventory (SPI), and the Athenstaedt Gender-Role Self-Concept scale.
- **OpenPsychometrics.org** (18 scales): HSQ, Taylor Manifest Anxiety, HEXACO, RIASEC, Consideration of Future Consequences, DASS, ECR, Empathizing/Systemizing Quotient, GCBS, KIMS, MACH-IV, MGKT, NPAS, Rosenberg Self-Esteem, RWAS, Sexual Compulsivity, AMBI, and the 16PF.
- **OSF & open research repositories** (8 scales): PID-5, SCL-90-R, Psychological Strain Scales, the Comprehensive Autistic Inventory / ASRS (plus an Adult ADHD self-report), C-PETS, EPTEPS, the Vanity scale, and a Self-Efficacy Fragility scale from an own study deposited to OSF.
- **Journal supplement (DOI)** (1 scale): General Attitudes towards AI Scale (GAAIS).

The two out-of-training splits come from separate studies kept fully disjoint from the training scales: the **Bainbridge** personality mega-study (holdout — item-disjoint, overlapping scale families) and a **SurveyBot** study of entirely new scales (validation — AAID + BFI-10).

## 🧪 Retired stages

Built, benchmarked, then removed because they didn't improve the Siamese model over raw embeddings + `global_sim`:

- **Cross-encoder features** — per-pair NLI (entail/contradict/neutral), reranker similarity, pair sentiment, and derived interactions. Permutation importance: ≈0.005 Pearson r over embeddings alone (vs. ≈0.85 when the embedding pathway itself is permuted).
- **Per-item sentiment/emotion** — 3-class sentiment, 7-class emotion classification per item. Same ablation result.
- **Autoencoder compression** — a 4096→512 bottleneck used to derive `global_sim` in a compressed space. `global_sim` is now raw cosine similarity computed directly on the full embedding, no compression step.

## 🗺️ Repository layout

```
pyproject.toml              # pip install -e . -- installs `pair` (inference only, no training data)
README.md · DEV_HISTORY.md · scale_sources.md
performance_logbook.xlsx    # one row per model_validation.py run

code/
├── __init__.py              # pair.predict / pair.embed / pair.to_matrix
├── inference.py             # the pip package: one-step predict(), standalone embed()
├── run_pipeline.ipynb       # thin notebook wrapper around train()
├── data_prep/
│   ├── data_integration.R
│   ├── embed_items.py
│   └── helper_functions/embedding_functions.py   # hf / ollama / api embedding backends
└── modelling/
    ├── siamese_model.py             # shared SiameseDNN architecture
    ├── model_training.py
    ├── model_validation.py
    ├── train.py                     # train() -- repo-only orchestration, not in the pip package
    ├── order_sensitivity_check.ipynb
    └── optuna_hpt/                  # standalone Optuna hyperparameter search

data/
├── scales/                  raw per-scale data + codebooks (see scale_sources.md)
├── raw/                     integrated train/holdout/validation parquet + per-backbone embedding caches
├── processed/                intermediate joined tables
└── clustered_embeddings/    legacy autoencoder-bottleneck tables (retired stage, kept for reference)

models/<backbone>/           dnn_siamese_cor.pt, quantile_transformer.joblib + _meta.json,
                              embedding_meta.json  -- mostly not committed to git, see "Model weights" above
```

## 💻 Hardware & hyperparameter tuning

Runs on CUDA GPUs (falls back to CPU); float32 throughout, with the full training-item embedding table held as a frozen on-device lookup so each batch only moves `(idx1, idx2, global_sim, target)` rows rather than the embeddings themselves.

Hyperparameters were searched with a standalone Optuna script (`code/modelling/optuna_hpt/dnn_siamese_holdout_tuning.py`, TPESampler + MedianPruner, multi-worker SQLite study) across four rounds as the feature set and search space evolved (`dnn_siamese_holdout_v1..v4_best.json`). The current production config (`encoder_dims=(384,)`, `head_dims=(256, 130)`, dropout 0.226, lr 2e-4, weight decay 0.0023, Huber β 0.127) is v3's; a v4 run (768-unit encoder, (384, 216) head) scores comparably but hasn't been adopted as the default yet.

## 🛣️ Further plans

- Distribute trained checkpoints without bloating git history (Git LFS, or hosting `models/` externally).
- Re-run Optuna HPT with v4's wider search space as the new baseline.