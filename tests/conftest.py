import os
from types import SimpleNamespace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from scunveil._inference import scUnveil
from scunveil._layers import PCAProjection


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

    inputs = tf.keras.Input(shape=(n_genes,))
    embeddings = tf.keras.layers.Dense(
        emb_dim,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=11),
        bias_initializer="zeros",
    )(inputs)
    embeddings = tf.keras.layers.LayerNormalization()(embeddings)
    outputs = tf.keras.layers.Dense(
        n_genes,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=13),
        bias_initializer="zeros",
    )(embeddings)

    model.full_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.raw_embedder = tf.keras.Model(inputs=inputs, outputs=embeddings)
    model.output_layer = model.full_model.layers[-1]
    model.expression_predictor = tf.keras.Sequential([model.output_layer])

    model.pca_mean = np.zeros((1, emb_dim), dtype=np.float32)
    model.pca_mat = np.eye(emb_dim, dtype=np.float32)
    model.pca_projector = tf.keras.Sequential([PCAProjection()])
    model.pca_projector.build((None, emb_dim))
    model.pca_projector.set_weights([model.pca_mean, model.pca_mat])

    model.input_anndata = None
    model.raw_embeddings = None
    model.pca_embeddings = None
    model.var_map_matrix = None
    model.gene_mapping_summary = None
    model._input_obs = None
    return model


@pytest.fixture
def tiny_model():
    return make_test_model()
