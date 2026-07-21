import os
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from scunveil import scUnveil


RUN_INTEGRATION = os.environ.get("SCUNVEIL_RUN_INTEGRATION") == "1"
TEST_DATA = Path(__file__).parents[1] / "test_shard.h5ad"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION or not TEST_DATA.is_file(),
    reason=(
        "Set SCUNVEIL_RUN_INTEGRATION=1 and provide test_shard.h5ad to run "
        "the real-checkpoint integration test."
    ),
)


def test_stable_checkpoint_end_to_end():
    backed = ad.read_h5ad(TEST_DATA, backed="r")
    try:
        input_adata = backed[:4, :].to_memory()
    finally:
        backed.file.close()

    model = scUnveil(verbose=False)
    model.set_input_anndata(input_adata, batch_size=2)

    assert model.gene_mapping_summary["mapped_input_features"] == 60_000
    assert model.get_raw_embeddings().shape == (4, 2_048)
    assert model.get_embeddings(16).shape == (4, 16)

    selected = model.get_specific_genes_imputation(
        ["CD4", "ENSG00000153563"], batch_size=2
    )
    assert selected.shape == (4, 2)
    assert selected.var["feature_name"].tolist() == ["CD4", "CD8A"]
    assert np.isfinite(selected.X).all()

    genes = model.get_genes_embeddings(normalize=False)
    expected = model.output_layer.kernel.numpy().T.astype(np.float16)
    np.testing.assert_array_equal(genes.X, expected)

    generated = model.generate_cells(3, 4, batch_size=2, seed=7)
    np.testing.assert_array_equal(
        np.asarray(generated.X.sum(axis=1)).ravel(), np.full(3, 4)
    )
