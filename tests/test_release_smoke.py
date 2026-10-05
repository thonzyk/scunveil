"""Run explicitly against the real default checkpoint in an installed package."""

import os

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from scunveil import scUnveil
from scunveil._inference import DEFAULT_MODEL_VERSION


@pytest.mark.skipif(
    os.environ.get("SCUNVEIL_RUN_RELEASE_SMOKE") != "1",
    reason="Set SCUNVEIL_RUN_RELEASE_SMOKE=1 to download and check the real checkpoint.",
)
def test_installed_package_downloads_default_checkpoint_and_predicts():
    model = scUnveil(verbose=False)
    assert model.model_version == DEFAULT_MODEL_VERSION
    assert model.config.n_genes == 60_000
    assert model.config.ff_dim == 2_048

    genes = model.reference_var.iloc[:100]
    counts = csr_matrix(([10, 5], ([0, 0], [0, 1])), shape=(1, 100))
    input_adata = AnnData(
        X=counts,
        obs=pd.DataFrame(index=["cell0"]),
        var=pd.DataFrame(
            {"feature_id": genes["feature_id"].to_numpy()},
            index=genes["feature_id"].to_numpy(),
        ),
    )
    result = model.process_anndata(input_adata, batch_size=1)

    raw = result.get_raw_embeddings()
    pca = result.get_embeddings(8)
    imputed = result.get_specific_genes_imputation(genes.iloc[0]["feature_id"])
    assert raw.shape == (1, 2_048)
    assert pca.shape == (1, 8)
    assert imputed.shape == (1, 1)
    assert np.isfinite(raw).all()
    assert np.isfinite(pca).all()
    assert np.isfinite(imputed.X).all()
