import os
from types import SimpleNamespace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from scunveil._inference import scUnveil
from scunveil._layers import PCAProjection
from scunveil._model import RNABagModel


def make_test_model(n_genes=8, emb_dim=4, reference_var=None):
    """Construct the inference interface around a tiny in-memory Keras model."""
    model = scUnveil.__new__(scUnveil)
    model.verbose = False
    model.model_version = "test"
    model.checkpoint_path = None
    model.config = SimpleNamespace(
        n_genes=n_genes,
        n_layers=1,
        emb_dim=emb_dim,
    )

    if reference_var is None:
        reference_var = pd.DataFrame(
            {
                "feature_id": [f"ENSG{i:011d}" for i in range(n_genes)],
                "feature_name": [f"GENE{i}" for i in range(n_genes)],
            }
        )
    model.reference_var = reference_var.reset_index(drop=True)
    model._build_reference_lookups()

    model.full_model = RNABagModel(n_vars=n_genes, n_layers=1, emb_dim=emb_dim).model
    model.raw_embedder = tf.keras.Model(
        model.full_model.input,
        model.full_model.get_layer('out_emb').output,
    )
    model.output_layer = model.full_model.get_layer('out_logits')
    model.expression_predictor = tf.keras.Sequential([model.output_layer])

    model.pca_mean = np.zeros((1, emb_dim), dtype=np.float32)
    model.pca_mat = np.eye(emb_dim, dtype=np.float32)
    model.pca_projector = tf.keras.Sequential([PCAProjection()])
    model.pca_projector.build((None, emb_dim))
    model.pca_projector.set_weights([model.pca_mean, model.pca_mat])

    return model


@pytest.fixture
def tiny_model():
    return make_test_model()
