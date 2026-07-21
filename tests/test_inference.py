import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from scipy.sparse import csc_matrix, csr_matrix

from conftest import make_test_model


def make_adata(matrix, ids=None, symbols=None, obs_names=None):
    matrix = matrix.copy()
    n_cells, n_genes = matrix.shape
    if ids is None:
        ids = [f"ENSG{i:011d}" for i in range(n_genes)]
    var = pd.DataFrame(
        {"feature_id": ids},
        index=[f"var{i}" for i in range(n_genes)],
    )
    if symbols is not None:
        var["feature_name"] = symbols
    if obs_names is None:
        obs_names = [f"cell{i}" for i in range(n_cells)]
    obs = pd.DataFrame(index=obs_names)
    return AnnData(X=matrix, obs=obs, var=var)


@pytest.mark.parametrize(
    "matrix_factory",
    [
        lambda x: x,
        csr_matrix,
        csc_matrix,
    ],
)
def test_dense_and_sparse_counts_are_supported(tiny_model, matrix_factory):
    counts = np.array(
        [[2, 0, 1, 0, 0, 3, 0, 1], [0, 1, 0, 2, 1, 0, 1, 0]],
        dtype=np.int32,
    )
    adata = make_adata(matrix_factory(counts))

    tiny_model.set_input_anndata(adata, batch_size=1)

    assert tiny_model.get_raw_embeddings().shape == (2, 4)
    assert tiny_model.get_embeddings(None).shape == (2, 4)
    assert tiny_model.gene_mapping_summary["mapped_feature_fraction"] == 1.0
    assert tiny_model.gene_mapping_summary["mapped_umi_fraction"] == 1.0


def test_backed_sparse_counts_are_supported(tiny_model, tmp_path):
    counts = csr_matrix(np.eye(8, dtype=np.int32)[:2])
    path = tmp_path / "input.h5ad"
    make_adata(counts).write_h5ad(path)
    backed = read_h5ad(path, backed="r")
    try:
        tiny_model.set_input_anndata(backed, batch_size=1)
        assert tiny_model.get_embeddings(2).shape == (2, 2)
    finally:
        backed.file.close()


@pytest.mark.parametrize(
    "bad_value, message",
    [
        (0.5, "raw integer-like UMI counts"),
        (-1.0, "negative"),
        (np.nan, "NaN or infinite"),
        (np.inf, "NaN or infinite"),
    ],
)
def test_invalid_counts_are_rejected(tiny_model, bad_value, message):
    counts = np.ones((2, 8), dtype=np.float64)
    counts[0, 0] = bad_value

    with pytest.raises(ValueError, match=message):
        tiny_model.set_input_anndata(make_adata(counts), batch_size=1)


def test_boolean_counts_are_rejected(tiny_model):
    with pytest.raises(TypeError, match="numeric UMI counts"):
        tiny_model.set_input_anndata(
            make_adata(csr_matrix(np.ones((2, 8), dtype=bool)))
        )


def test_versioned_ensembl_ids_are_supported(tiny_model):
    counts = csr_matrix(np.ones((2, 8), dtype=np.int32))
    ids = [f"ENSG{i:011d}.{i + 1}" for i in range(8)]

    tiny_model.set_input_anndata(make_adata(counts, ids=ids), batch_size=2)

    assert tiny_model.gene_mapping_summary["identifier_type"] == "feature_id"
    assert tiny_model.gene_mapping_summary["mapped_input_features"] == 8


def test_detection_uses_the_complete_reference_not_first_100_genes():
    n_genes = 120
    reference = pd.DataFrame(
        {
            "feature_id": [f"ENSG{i:011d}" for i in range(n_genes)],
            "feature_name": [f"GENE{i}" for i in range(n_genes)],
        }
    )
    model = make_test_model(
        n_genes=n_genes,
        emb_dim=3,
        reference_var=reference,
    )
    late_symbols = [f"GENE{i}" for i in range(100, 120)]
    adata = AnnData(
        X=csr_matrix(np.ones((2, 20), dtype=np.int32)),
        var=pd.DataFrame(index=late_symbols),
    )

    model.set_input_anndata(adata, batch_size=2)

    assert model.gene_mapping_summary["identifier_type"] == "feature_name"
    assert model.gene_mapping_summary["mapped_input_features"] == 20


def test_low_feature_mapping_is_rejected(tiny_model):
    ids = ["ENSG00000000000", "ENSG00000000001"] + [f"UNKNOWN{i}" for i in range(6)]
    counts = csr_matrix(np.ones((2, 8), dtype=np.int32))

    with pytest.raises(ValueError, match="25.0%"):
        tiny_model.set_input_anndata(make_adata(counts, ids=ids))


def test_low_umi_mapping_is_rejected(tiny_model):
    ids = [f"ENSG{i:011d}" for i in range(4)] + [f"UNKNOWN{i}" for i in range(4)]
    counts = np.zeros((2, 8), dtype=np.int32)
    counts[:, 4:] = 10

    with pytest.raises(ValueError, match="0.0% of input UMI counts"):
        tiny_model.set_input_anndata(make_adata(csr_matrix(counts), ids=ids))


def test_duplicate_input_mapping_is_rejected(tiny_model):
    ids = [f"ENSG{i:011d}" for i in range(8)]
    ids[-1] = ids[0]
    counts = csr_matrix(np.ones((2, 8), dtype=np.int32))

    with pytest.raises(ValueError, match="Multiple input features"):
        tiny_model.set_input_anndata(make_adata(counts, ids=ids))


def test_ambiguous_symbols_are_not_arbitrarily_mapped():
    reference = pd.DataFrame(
        {
            "feature_id": [f"ENSG{i:011d}" for i in range(8)],
            "feature_name": ["A", "B", "C", "D", "E", "F", "DUP", "DUP"],
        }
    )
    model = make_test_model(reference_var=reference)
    adata = AnnData(
        X=csr_matrix(np.ones((2, 8), dtype=np.int32)),
        var=pd.DataFrame(index=["A", "B", "C", "D", "E", "F", "X", "Y"]),
    )
    model.set_input_anndata(adata)

    with pytest.raises(ValueError, match="multiple model genes"):
        model.get_specific_genes_imputation("DUP")


def test_failed_input_replacement_preserves_previous_state(tiny_model):
    valid = make_adata(csr_matrix(np.ones((2, 8), dtype=np.int32)))
    tiny_model.set_input_anndata(valid)
    previous_embeddings = tiny_model.get_embeddings(None)
    previous_summary = tiny_model.gene_mapping_summary.copy()

    invalid = make_adata(
        csr_matrix(np.ones((3, 8), dtype=np.int32)),
        ids=[f"UNKNOWN{i}" for i in range(8)],
        obs_names=["new0", "new1", "new2"],
    )
    with pytest.raises(ValueError):
        tiny_model.set_input_anndata(invalid)

    assert tiny_model.input_anndata is valid
    np.testing.assert_array_equal(tiny_model.get_embeddings(None), previous_embeddings)
    assert tiny_model.gene_mapping_summary == previous_summary


def test_public_results_require_an_input(tiny_model):
    with pytest.raises(RuntimeError, match="set_input_anndata"):
        tiny_model.get_raw_embeddings()
    with pytest.raises(RuntimeError, match="set_input_anndata"):
        tiny_model.get_embeddings()
    with pytest.raises(RuntimeError, match="set_input_anndata"):
        tiny_model.get_all_genes_imputation()


@pytest.mark.parametrize("n_features", [0, -1, 5, 1.5, True])
def test_embedding_dimension_is_validated(tiny_model, n_features):
    tiny_model.set_input_anndata(
        make_adata(csr_matrix(np.ones((2, 8), dtype=np.int32)))
    )
    expected_exception = TypeError if n_features in (1.5, True) else ValueError
    with pytest.raises(expected_exception):
        tiny_model.get_embeddings(n_features)


def test_selected_imputation_matches_all_gene_columns(tiny_model):
    counts = csr_matrix(
        np.array(
            [[2, 0, 1, 0, 4, 0, 1, 0], [0, 2, 0, 3, 0, 1, 0, 1]],
            dtype=np.int32,
        )
    )
    tiny_model.set_input_anndata(make_adata(counts), batch_size=1)

    all_genes = tiny_model.get_all_genes_imputation(batch_size=1)
    selected = tiny_model.get_specific_genes_imputation(
        ["GENE3", "ENSG00000000001.9"], batch_size=1
    )

    np.testing.assert_array_equal(selected.X, all_genes.X[:, [3, 1]])
    assert selected.var["feature_name"].tolist() == ["GENE3", "GENE1"]
    assert np.isfinite(selected.X).all()


def test_gene_embeddings_are_the_transposed_output_kernel(tiny_model):
    result = tiny_model.get_genes_embeddings(normalize=False)
    expected = tiny_model.output_layer.kernel.numpy().T.astype(np.float16)

    np.testing.assert_array_equal(result.X, expected)
    assert result.shape == (8, 4)
    assert result.obs_names[0] == "ENSG00000000000"


def test_generated_cells_have_requested_depth_and_seed(tiny_model):
    first = tiny_model.generate_cells(5, 7, batch_size=2, seed=123)
    second = tiny_model.generate_cells(5, 7, batch_size=2, seed=123)

    np.testing.assert_array_equal(first.X.toarray(), second.X.toarray())
    np.testing.assert_array_equal(
        np.asarray(first.X.sum(axis=1)).ravel(), np.full(5, 7)
    )


def test_fully_enriched_h5ad_aligns_all_axes(tiny_model):
    tiny_model.set_input_anndata(
        make_adata(csr_matrix(np.ones((3, 8), dtype=np.int32)))
    )

    enriched = tiny_model.get_fully_enriched_h5ad(
        batch_size=2,
        list_of_genes=["GENE4", "GENE1"],
        n_embedding_features=2,
    )

    assert enriched.shape == (3, 2)
    assert enriched.var["feature_name"].tolist() == ["GENE4", "GENE1"]
    assert enriched.obsm["X_scunveil"].shape == (3, 2)
    assert enriched.varm["scunveil_gene_embeddings"].shape == (2, 4)
    all_gene_embeddings = tiny_model.get_genes_embeddings(normalize=True)
    np.testing.assert_array_equal(
        enriched.varm["scunveil_gene_embeddings"],
        all_gene_embeddings.X[[4, 1]],
    )
