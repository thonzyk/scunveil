import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from scipy.sparse import csc_matrix, csr_matrix

from conftest import make_test_model
from scunveil import ScUnveilResult


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

    result = tiny_model.process_anndata(adata, batch_size=1)

    assert result.get_raw_embeddings().shape == (2, 4)
    assert result.get_embeddings(None).shape == (2, 4)
    assert result.gene_mapping_summary["mapped_feature_fraction"] == 1.0
    assert result.gene_mapping_summary["mapped_umi_fraction"] == 1.0


def test_backed_sparse_counts_are_supported(tiny_model, tmp_path):
    counts = csr_matrix(np.eye(8, dtype=np.int32)[:2])
    path = tmp_path / "input.h5ad"
    make_adata(counts).write_h5ad(path)
    backed = read_h5ad(path, backed="r")
    try:
        result = tiny_model.process_anndata(backed, batch_size=1)
        assert result.get_embeddings(2).shape == (2, 2)
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
        tiny_model.process_anndata(make_adata(counts), batch_size=1)


def test_boolean_counts_are_rejected(tiny_model):
    with pytest.raises(TypeError, match="numeric UMI counts"):
        tiny_model.process_anndata(
            make_adata(csr_matrix(np.ones((2, 8), dtype=bool)))
        )


def test_versioned_ensembl_ids_are_supported(tiny_model):
    counts = csr_matrix(np.ones((2, 8), dtype=np.int32))
    ids = [f"ENSG{i:011d}.{i + 1}" for i in range(8)]

    result = tiny_model.process_anndata(make_adata(counts, ids=ids), batch_size=2)

    assert result.gene_mapping_summary["identifier_type"] == "feature_id"
    assert result.gene_mapping_summary["mapped_input_features"] == 8


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

    result = model.process_anndata(adata, batch_size=2)

    assert result.gene_mapping_summary["identifier_type"] == "feature_name"
    assert result.gene_mapping_summary["mapped_input_features"] == 20


def test_low_feature_mapping_is_rejected(tiny_model):
    ids = ["ENSG00000000000", "ENSG00000000001"] + [f"UNKNOWN{i}" for i in range(6)]
    counts = csr_matrix(np.ones((2, 8), dtype=np.int32))

    with pytest.raises(ValueError, match="25.0%"):
        tiny_model.process_anndata(make_adata(counts, ids=ids))


def test_low_umi_mapping_is_rejected(tiny_model):
    ids = [f"ENSG{i:011d}" for i in range(4)] + [f"UNKNOWN{i}" for i in range(4)]
    counts = np.zeros((2, 8), dtype=np.int32)
    counts[:, 4:] = 10

    with pytest.raises(ValueError, match="0.0% of input UMI counts"):
        tiny_model.process_anndata(make_adata(csr_matrix(counts), ids=ids))


def test_duplicate_input_mapping_is_rejected(tiny_model):
    ids = [f"ENSG{i:011d}" for i in range(8)]
    ids[-1] = ids[0]
    counts = csr_matrix(np.ones((2, 8), dtype=np.int32))

    with pytest.raises(ValueError, match="Multiple input features"):
        tiny_model.process_anndata(make_adata(counts, ids=ids))


@pytest.mark.parametrize("duplicate_ids", [
    [f"ENSG{i:011d}" for i in range(7)] + ["ENSG00000000000"],
    [f"ENSG{i:011d}" for i in range(7)] + ["ENSG00000000000.2"],
])
def test_duplicate_best_column_falls_back_to_valid_symbols(tiny_model, duplicate_ids):
    counts = csr_matrix(np.ones((2, 8), dtype=np.int32))
    symbols = [f"GENE{i}" for i in range(6)] + ["UNKNOWN6", "UNKNOWN7"]
    adata = make_adata(counts, ids=duplicate_ids, symbols=symbols)

    result = tiny_model.process_anndata(adata)

    assert result.gene_mapping_summary["identifier_column"] == "feature_name"
    assert result.gene_mapping_summary["mapped_input_features"] == 6
    assert result.gene_mapping_summary["mapped_umi_fraction"] == 0.75
    np.testing.assert_array_equal(result.var_map_matrix.toarray()[7], np.zeros(8))


def test_low_umi_best_column_falls_back_to_better_symbols(tiny_model):
    ids = [f"ENSG{i:011d}" for i in range(6)] + ["UNKNOWN6", "UNKNOWN7"]
    symbols = [f"UNKNOWN{i}" for i in range(3)] + [f"GENE{i}" for i in range(3, 8)]
    counts = csr_matrix([[1, 1, 1, 1, 1, 1, 20, 20]])

    result = tiny_model.process_anndata(make_adata(counts, ids=ids, symbols=symbols))

    assert result.gene_mapping_summary["identifier_column"] == "feature_name"
    assert result.gene_mapping_summary["mapped_input_features"] == 5
    assert result.gene_mapping_summary["mapped_umi_fraction"] == pytest.approx(43 / 46)


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
    result = model.process_anndata(adata)

    with pytest.raises(ValueError, match="multiple model genes"):
        result.get_specific_genes_imputation("DUP")


def test_failed_processing_preserves_previous_result(tiny_model):
    valid = make_adata(csr_matrix(np.ones((2, 8), dtype=np.int32)))
    result = tiny_model.process_anndata(valid)
    previous_embeddings = result.get_embeddings(None)
    previous_summary = result.gene_mapping_summary.copy()

    invalid = make_adata(
        csr_matrix(np.ones((3, 8), dtype=np.int32)),
        ids=[f"UNKNOWN{i}" for i in range(8)],
        obs_names=["new0", "new1", "new2"],
    )
    with pytest.raises(ValueError):
        tiny_model.process_anndata(invalid)

    assert result.input_anndata is valid
    np.testing.assert_array_equal(result.get_embeddings(None), previous_embeddings)
    assert result.gene_mapping_summary == previous_summary


def test_two_results_share_model_and_keep_independent_data(tiny_model):
    first_input = make_adata(csr_matrix([[2, 0, 0, 0, 0, 0, 0, 0]]))
    second_input = make_adata(csr_matrix([[0, 0, 3, 0, 0, 0, 0, 0]]))

    first = tiny_model.process_anndata(first_input)
    first_embeddings = first.get_embeddings(None)
    first_imputation = first.get_specific_genes_imputation("GENE0").X.copy()
    second = tiny_model.process_anndata(second_input)

    assert isinstance(first, ScUnveilResult)
    assert first.sc_unveil is second.sc_unveil is tiny_model
    assert first.input_anndata is first_input
    assert second.input_anndata is second_input
    assert first.raw_embeddings is not second.raw_embeddings
    assert first.var_map_matrix is not second.var_map_matrix
    assert not hasattr(tiny_model, "input_anndata")
    np.testing.assert_array_equal(first.get_embeddings(None), first_embeddings)
    np.testing.assert_array_equal(
        first.get_specific_genes_imputation("GENE0").X, first_imputation
    )
    assert second.get_embeddings(None).shape == first_embeddings.shape
    assert not np.array_equal(second.get_embeddings(None), first_embeddings)
    assert second.get_specific_genes_imputation("GENE0").shape == (1, 1)


@pytest.mark.parametrize("n_features", [0, -1, 5, 1.5, True])
def test_embedding_dimension_is_validated(tiny_model, n_features):
    result = tiny_model.process_anndata(
        make_adata(csr_matrix(np.ones((2, 8), dtype=np.int32)))
    )
    expected_exception = TypeError if n_features in (1.5, True) else ValueError
    with pytest.raises(expected_exception):
        result.get_embeddings(n_features)


def test_selected_imputation_matches_all_gene_columns(tiny_model):
    counts = csr_matrix(
        np.array(
            [[2, 0, 1, 0, 4, 0, 1, 0], [0, 2, 0, 3, 0, 1, 0, 1]],
            dtype=np.int32,
        )
    )
    result = tiny_model.process_anndata(make_adata(counts), batch_size=1)

    all_genes = result.get_all_genes_imputation(batch_size=1)
    selected = result.get_specific_genes_imputation(
        ["GENE3", "ENSG00000000001.9"], batch_size=1
    )

    np.testing.assert_array_equal(selected.X, all_genes.X[:, [3, 1]])
    assert selected.var["feature_name"].tolist() == ["GENE3", "GENE1"]
    assert np.isfinite(selected.X).all()


def test_gene_embeddings_are_the_shared_gene_matrix(tiny_model):
    result = tiny_model.get_genes_embeddings(normalize=False)
    expected = tiny_model.output_layer.input_projection.kernel.numpy().astype(np.float16)

    np.testing.assert_array_equal(result.X, expected)
    assert result.shape == (8, 4)
    assert result.obs_names[0] == "ENSG00000000000"


@pytest.mark.parametrize(
    "genes, indices",
    [
        (None, list(range(8))),
        (["GENE3", "ENSG00000000001.9", "GENE6"], [3, 1, 6]),
    ],
)
def test_finetuning_clone_uses_raw_counts_and_independent_weights(
    tiny_model, genes, indices
):
    clone = tiny_model.get_model_clone_for_finetuning(
        output_dim=3, list_of_input_genes=genes
    )
    counts = np.arange(1, len(indices) + 1, dtype=np.float32)[None, :]
    full_counts = np.zeros((1, 8), dtype=np.float32)
    full_counts[:, indices] = counts

    original_embedding = tiny_model.raw_embedder(np.log1p(full_counts), training=False)
    expected = clone.layers[-1](original_embedding)
    np.testing.assert_allclose(clone(counts, training=False), expected, atol=1e-6)
    assert clone.input_shape == (None, len(indices))
    assert clone.output_shape == (None, 3)

    original_kernel = tiny_model.output_layer.input_projection.kernel
    before = original_kernel.numpy().copy()
    backbone = clone.get_layer(tiny_model.raw_embedder.name)
    clone_kernel = backbone.get_layer(
        tiny_model.output_layer.input_projection.name
    ).kernel
    clone_kernel.assign(clone_kernel.numpy() + 1)
    np.testing.assert_array_equal(original_kernel.numpy(), before)


def test_finetuning_clone_rejects_invalid_genes(tiny_model):
    with pytest.raises(ValueError, match="output_dim"):
        tiny_model.get_model_clone_for_finetuning(0)
    with pytest.raises(ValueError, match="not present"):
        tiny_model.get_model_clone_for_finetuning(2, ["UNKNOWN"])
    with pytest.raises(ValueError, match="same model gene"):
        tiny_model.get_model_clone_for_finetuning(
            2, ["GENE1", "ENSG00000000001"]
        )


def test_generated_cells_have_requested_depth_and_seed(tiny_model):
    first = tiny_model.generate_cells(5, 7, batch_size=2, seed=123)
    second = tiny_model.generate_cells(5, 7, batch_size=2, seed=123)

    np.testing.assert_array_equal(first.X.toarray(), second.X.toarray())
    np.testing.assert_array_equal(
        np.asarray(first.X.sum(axis=1)).ravel(), np.full(5, 7)
    )


def test_seeded_generation_explains_batch_size_dependency(tiny_model, capsys):
    tiny_model.verbose = True
    tiny_model.generate_cells(1, 1, batch_size=1, seed=123)
    assert "same seed and batch size" in capsys.readouterr().out


def test_large_allocation_warning_uses_shape_without_allocating(tiny_model, monkeypatch):
    import scunveil._inference as inference

    monkeypatch.setattr(inference, "LARGE_ALLOCATION_BYTES", 16)
    with pytest.warns(UserWarning, match="largest expected single allocation.*peak memory"):
        tiny_model._warn_large_allocation("Example", (4, 2), np.float32)


def test_generation_warns_before_large_allocation(tiny_model, monkeypatch):
    import scunveil._inference as inference

    monkeypatch.setattr(inference, "LARGE_ALLOCATION_BYTES", 1)
    with pytest.warns(UserWarning, match="Generating cells"):
        tiny_model.generate_cells(1, 1)


def test_processing_and_imputation_warn_before_large_allocations(tiny_model, monkeypatch):
    import scunveil._inference as inference

    monkeypatch.setattr(inference, "LARGE_ALLOCATION_BYTES", 1)
    with pytest.warns(UserWarning, match="Processing cells"):
        result = tiny_model.process_anndata(
            make_adata(csr_matrix(np.ones((2, 8), dtype=np.int32)))
        )
    with pytest.warns(UserWarning, match="All-gene imputation"):
        result.get_all_genes_imputation()


def test_fully_enriched_h5ad_aligns_all_axes(tiny_model):
    result = tiny_model.process_anndata(
        make_adata(csr_matrix(np.ones((3, 8), dtype=np.int32)))
    )

    enriched = result.get_fully_enriched_h5ad(
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


def test_fully_enriched_h5ad_accepts_one_shot_gene_iterable(tiny_model):
    result = tiny_model.process_anndata(
        make_adata(csr_matrix(np.ones((2, 8), dtype=np.int32)))
    )
    genes = (gene for gene in ["GENE4", "GENE1"])

    enriched = result.get_fully_enriched_h5ad(list_of_genes=genes)

    assert enriched.var["feature_name"].tolist() == ["GENE4", "GENE1"]
    assert enriched.shape == (2, 2)
