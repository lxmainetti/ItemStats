"""PAIR -- Predicting Associations of Inter-item Relationships.

Inference-only package: embeds item text and scores pairs with a trained
Siamese DNN. No training data or checkpoints ship with this package --
predict() reads them straight from a PAIR repo's data/raw/ + models/
folders (same model_safe names as the training pipeline), default: the repo
this package was pip install -e'd from.

    import pair
    df = pair.predict(["item text one", "item text two", "item text three"])
    # model="Qwen/Qwen3-Embedding-8B" by default -- pass model=<other HF/Ollama
    # id, or model_safe folder name> for a different trained backbone, or
    # models_root= for a different repo checkout.

To train your own backbone (or reproduce/extend the results), clone the full
repo from GitHub and run code/modelling/train.py or code/run_pipeline.ipynb
directly -- that needs the training data pipeline, which this pip package
doesn't carry.
"""

from .inference import predict_correlations as predict, to_matrix, embed

__all__ = ["predict", "to_matrix", "embed"]
