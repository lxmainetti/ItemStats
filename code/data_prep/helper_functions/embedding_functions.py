"""Embedding backends.

Logic: Two helper functions that turn ``item_list["item"]`` into a list of numeric
       embedding vectors, either through a local Ollama model or through the
       OpenAI / Google embedding API.
Workflow: Called by ``embed_items.ipynb``; the local backend writes the raw
          embedding list to ``data/raw/<model_safe>/embeddings_raw.pkl`` as a
          side-effect.

Each function returns ``(embeddings, model_safe)`` so the caller gets the
filename-safe model id back (the R version shared this via ``<<-``).
"""

import os
import time

import numpy as np
import requests


# ---- Local Ollama embeddings ----

def get_embeddings(item_list, model="qwen3-embedding:8b", include_instruction=True):
    """Generate embeddings using a locally-served Ollama model.

    ``qwen3-embedding:8b`` with the task-specific instruction prefix is the best
    performer on the inter-item correlation task in my benchmarks.

    Parameters
    ----------
    item_list : polars.DataFrame
        Must contain an ``item`` column.

    Returns
    -------
    (list[numpy.ndarray], str)
        The embedding vectors and the filename-safe model id.
    """
    import ollama
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm

    # Filename-safe model identifier returned to the caller
    model_safe = model.replace(":", "-").replace("/", "")

    # Override the path label when running qwen3 with the instruction prefix so
    # instructed and non-instructed runs don't overwrite each other on disk
    instruct_model = ("qwen3" in model) and include_instruction
    if instruct_model:
        model_safe = "qwen3-8b-with-instruction"

    # Build the model input: prepend Qwen3's recommended instruction prefix if
    # we're using the instructed variant; otherwise feed the raw item text
    if instruct_model:
        task_desc = (
            "Given a psychometric scale item, represent its latent psychological "
            "constructs for predicting response correlations and factor structures."
        )
        inputs = [f"Instruct: {task_desc}\nQuery: {item}" for item in item_list["item"]]
    else:
        inputs = list(item_list["item"])

    # Embed in parallel via Ollama, save raw response to disk for re-use
    def _embed_one(x):
        resp = ollama.embed(model=model, input=x)
        return np.asarray(resp["embeddings"][0], dtype=float)

    with ThreadPoolExecutor(max_workers=16) as pool:
        embeddings_list = list(tqdm(pool.map(_embed_one, inputs), total=len(inputs)))

    return embeddings_list, model_safe


# ---- Hosted-API embeddings (OpenAI / Google) ----

# Known native output size per OpenAI model, used when dims isn't given
# explicitly. text-embedding-3-small maxes out at 1536; requesting more (e.g.
# the 3072 that's valid for -large) gets a 400 from the API.
_OPENAI_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def get_embeddings_API(item_list, model="text-embedding-3-large", provider="openai",
                       dims=None, task_type="SEMANTIC_SIMILARITY"):
    """Generate embeddings using OpenAI or Gemini.

    Provider is detected from ``provider``. Output dimensionality is fixed for
    older OpenAI models, tunable from ``text-embedding-3-*`` onwards, and
    configurable on Gemini. If ``dims`` is left as ``None``, it's resolved from
    ``_OPENAI_MODEL_DIMS`` for known OpenAI models (falls back to 3072 for
    anything else, e.g. Gemini models, matching the previous fixed default).

    Returns
    -------
    (dict[str, numpy.ndarray], str)
        Embeddings keyed by item text, and the filename-safe model id.
    """
    items = list(item_list["item"])
    model_safe = model.replace(":", "-").replace("/", "-")

    if dims is None:
        dims = _OPENAI_MODEL_DIMS.get(model, 3072)

    print(f"Starting {provider} embedding for {len(items)} items...")
    embeddings_list = []

    def _post_with_retry(url, *, json_body, headers=None, params=None, max_tries=3):
        last_err = None
        for _ in range(max_tries):
            resp = requests.post(url, json=json_body, headers=headers, params=params)
            if resp.status_code < 400:
                return resp
            # 4xx = the request itself is wrong (bad model/dims/key/etc.) -- retrying
            # identical bad input won't help, so fail immediately with the API's own
            # error message instead of burning attempts and hiding what went wrong.
            if resp.status_code < 500:
                raise requests.HTTPError(
                    f"{resp.status_code} Client Error: {resp.text}", response=resp
                )
            last_err = requests.HTTPError(f"{resp.status_code} Server Error: {resp.text}", response=resp)
        raise last_err

    # ---- OpenAI ----
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

        body_list = {"model": model}
        # Only embedding-3 generation supports tunable output dims
        if "-3-" in model:
            body_list["dimensions"] = dims

        # OpenAI accepts up to 2048 inputs per request; batch at 100 for safety
        item_batches = [items[i:i + 100] for i in range(0, len(items), 100)]

        for batch in item_batches:
            body = dict(body_list)
            body["input"] = batch

            resp = _post_with_retry(
                "https://api.openai.com/v1/embeddings",
                json_body=body,
                headers={"Authorization": f"Bearer {api_key}"},
            )

            res_data = resp.json()
            batch_results = [np.asarray(d["embedding"], dtype=float) for d in res_data["data"]]
            embeddings_list.extend(batch_results)

            time.sleep(0.3)  # rate-limit buffer

    # ---- Google (Gemini) ----
    if provider == "google":
        from concurrent.futures import ThreadPoolExecutor

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        task_pfx = "task: sentence similarity | query: "
        # Gemini batchEmbedContents tops out at 100 inputs per request
        batches = [items[i:i + 100] for i in range(0, len(items), 100)]

        def _embed_batch(b):
            body = {"requests": [{
                "model": f"models/{model}",
                "content": {"parts": [{"text": f"{task_pfx}{x}"}]},
                "outputDimensionality": dims,
            } for x in b]}
            resp = _post_with_retry(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents",
                json_body=body,
                params={"key": api_key},
            )
            return [np.asarray(e["values"], dtype=float) for e in resp.json()["embeddings"]]

        with ThreadPoolExecutor(max_workers=10) as pool:  # tune to your rate limit
            for batch_results in pool.map(_embed_batch, batches):
                embeddings_list.extend(batch_results)

    # Key by item text and persist
    embeddings_by_item = dict(zip(items, embeddings_list))
    print("SUCCESS")

    return embeddings_by_item, model_safe


# ---- Local HuggingFace embeddings (sentence-transformers) ----

# ---- Local HuggingFace embeddings (sentence-transformers) ----

def get_embeddings_HF(item_list, model="dwulff/mpnet-personality", instruction=None,
                      batch_size=32, max_seq_length=256, normalize=False, quantize=0):
    """Generate embeddings from a local HF / sentence-transformers model.

    Per-model quirks (instruction handling, EOS, padding, 4-bit) are set in the
    ``match`` below. Returns ``(list[numpy.ndarray], model_safe)``.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    psycho_instruct = (
        "Given a psychometric scale item, represent its latent psychological "
        "constructs for predicting inter-item response correlations."
    )

    # Per-model config: (quantize_4bit, instruction_mode, instruction, needs_eos, padding_side)
    match model:
        case "nvidia/NV-Embed-v2":
            instr_mode, instr, needs_eos, padding_side = (
                "prompt", instruction or psycho_instruct, True, "right")
        case "Alibaba-NLP/gte-Qwen2-7B-instruct":
            instr_mode, instr, needs_eos, padding_side = (
                "prompt", instruction or psycho_instruct, False, None)
        case "intfloat/e5-mistral-7b-instruct":
            instr_mode, instr, needs_eos, padding_side = (
                "prepend", f"Instruct: {instruction or psycho_instruct}\nQuery: ",
                False, None)
        case "tencent/KaLM-Embedding-Gemma3-12B-2511":
            instr_mode, instr, needs_eos, padding_side = (
                "prompt", instruction or psycho_instruct, False, None)
        case "Qwen/Qwen3-Embedding-8B":
            instr_mode, instr, needs_eos, padding_side = (
                "prompt", instruction or psycho_instruct, False, None)
        case "Qwen/Qwen3-Embedding-4B":
            instr_mode, instr, needs_eos, padding_side = (
                "prompt", instruction or psycho_instruct, False, None)
        case "intfloat/e5-large-v2":
            instr_mode, instr, needs_eos, padding_side = (
                "prepend", f"Instruct: {instruction or psycho_instruct}\nQuery: ",
                False, None)
        case _:
            print(f"[get_embeddings_HF] no case for '{model}', using plain defaults.")
            instr_mode, instr, needs_eos, padding_side = (
                "none", None, False, None)

    model_safe = model.replace(":", "-").replace("/", "-")

    if quantize > 0:
        model_safe = f"{model_safe}-{quantize}bit"

    items = list(item_list["item"])

    # ---- load (optionally 4-bit) ----
    model_kwargs = {}
    if quantize == 4:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantize == 8:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,              # outlier cutoff; 6.0 is the default
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16
        
    config_kwargs = {}
    
    if model == "nvidia/NV-Embed-v2":
        config_kwargs["use_cache"] = False
        model_kwargs["device_map"] = "cuda"

    st = SentenceTransformer(model, trust_remote_code=True, 
                             model_kwargs=model_kwargs or None,
                             config_kwargs=config_kwargs or None
                             )

    st.max_seq_length = max_seq_length
    if padding_side:
        st.tokenizer.padding_side = padding_side

    # ---- build inputs ----
    texts = list(items)
    if instr_mode == "prepend" and instr:
        texts = [f"{instr}{t}" for t in texts]
    if needs_eos:
        texts = [t + st.tokenizer.eos_token for t in texts]

    encode_kwargs = dict(batch_size=batch_size, normalize_embeddings=normalize,
                         show_progress_bar=True)
    if instr_mode == "prompt" and instr:
        encode_kwargs["prompt"] = f"Instruct: {instr}\nQuery: "

    emb = st.encode(texts, **encode_kwargs)
    embeddings_list = [np.asarray(v, dtype=float) for v in emb]

    return embeddings_list, model_safe