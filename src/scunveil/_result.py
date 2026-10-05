from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

if TYPE_CHECKING:
    from ._inference import scUnveil


class ScUnveilResult:
    """One processed dataset, linked to its shared scUNVEIL model.

    The input AnnData is retained by reference. Keep a backed input file open
    while accessing ``input_anndata`` through this result.
    """

    def __init__(
        self,
        sc_unveil: scUnveil,
        input_anndata: ad.AnnData,
        var_map_matrix: csr_matrix,
        raw_embeddings: np.ndarray,
        pca_embeddings: np.ndarray,
        gene_mapping_summary: dict[str, Any],
    ) -> None:
        self.sc_unveil = sc_unveil
        self.input_anndata = input_anndata
        self._input_obs = input_anndata.obs.copy()
        self.var_map_matrix = var_map_matrix
        self.raw_embeddings = raw_embeddings
        self.pca_embeddings = pca_embeddings
        self.gene_mapping_summary = gene_mapping_summary

    def get_raw_embeddings(self) -> np.ndarray:
        """Return the unrotated model embeddings."""
        self.sc_unveil._warn_large_allocation(
            "Raw embeddings", self.raw_embeddings.shape, self.raw_embeddings.dtype
        )
        return self.raw_embeddings.copy()

    def get_embeddings(self, n_features: Optional[int] = 512) -> np.ndarray:
        """Return PCA-ordered cell embeddings.

        ``None`` returns all PCA components. A positive integer returns that
        many leading components. Use :meth:`get_raw_embeddings` for the
        unrotated hidden state.
        """
        if n_features is None:
            self.sc_unveil._warn_large_allocation(
                "PCA embeddings", self.pca_embeddings.shape, self.pca_embeddings.dtype
            )
            return self.pca_embeddings.copy()

        n_features = self.sc_unveil._positive_integer(n_features, "n_features")
        if n_features > self.sc_unveil.config.emb_dim:
            raise ValueError(
                "n_features cannot exceed the model embedding dimension "
                f"({self.sc_unveil.config.emb_dim})."
            )
        selected = self.pca_embeddings[:, :n_features]
        self.sc_unveil._warn_large_allocation(
            "PCA embeddings", selected.shape, selected.dtype
        )
        return selected.copy()

    def get_all_genes_imputation(self, batch_size: int = 128) -> ad.AnnData:
        """Return imputed expression for all model genes as log10(CPM)."""
        batch_size = self.sc_unveil._positive_integer(batch_size, "batch_size")

        n_cells = self.raw_embeddings.shape[0]
        largest_bytes = max(
            n_cells * self.sc_unveil.config.n_genes * np.dtype(np.float16).itemsize,
            min(n_cells, batch_size) * self.sc_unveil.config.n_genes
            * np.dtype(np.float32).itemsize,
        )
        self.sc_unveil._warn_large_allocation(
            "All-gene imputation", (largest_bytes,), np.uint8
        )

        gene_expressions = np.zeros(
            (n_cells, self.sc_unveil.config.n_genes), dtype=np.float16
        )
        with tqdm(
            total=n_cells,
            disable=not self.sc_unveil.verbose,
            desc="scUNVEIL imputation",
            unit="cell",
        ) as progress:
            for start in range(0, n_cells, batch_size):
                raw_batch = self.raw_embeddings[start : start + batch_size]
                prediction = self.sc_unveil._predict_log10_cpm(raw_batch)
                n_batch = raw_batch.shape[0]
                gene_expressions[start : start + n_batch] = prediction
                progress.update(n_batch)

        return ad.AnnData(
            X=gene_expressions,
            obs=self._input_obs.copy(),
            var=self.sc_unveil._model_var(),
        )

    def get_specific_genes_imputation(
        self, list_of_gene_names: Union[str, Iterable[str]], batch_size: int = 128
    ) -> ad.AnnData:
        """Return selected-gene imputation in request order as log10(CPM)."""
        batch_size = self.sc_unveil._positive_integer(batch_size, "batch_size")
        _, gene_indices = self.sc_unveil._resolve_requested_genes(list_of_gene_names)

        n_cells = self.raw_embeddings.shape[0]
        largest_bytes = max(
            n_cells * len(gene_indices) * np.dtype(np.float16).itemsize,
            min(n_cells, batch_size) * self.sc_unveil.config.n_genes
            * np.dtype(np.float32).itemsize,
        )
        self.sc_unveil._warn_large_allocation(
            "Selected-gene imputation", (largest_bytes,), np.uint8
        )
        gene_expressions = np.zeros((n_cells, len(gene_indices)), dtype=np.float16)
        with tqdm(
            total=n_cells,
            disable=not self.sc_unveil.verbose,
            desc="scUNVEIL imputation",
            unit="cell",
        ) as progress:
            for start in range(0, n_cells, batch_size):
                raw_batch = self.raw_embeddings[start : start + batch_size]
                prediction = self.sc_unveil._predict_log10_cpm(raw_batch)
                selected_prediction = prediction[:, gene_indices]
                n_batch = raw_batch.shape[0]
                gene_expressions[start : start + n_batch] = selected_prediction
                progress.update(n_batch)

        return ad.AnnData(
            X=gene_expressions,
            obs=self._input_obs.copy(),
            var=self.sc_unveil._model_var(gene_indices),
        )

    def get_fully_enriched_h5ad(
        self,
        batch_size: int = 128,
        list_of_genes: Optional[Union[str, Iterable[str]]] = None,
        n_embedding_features: Optional[int] = None,
    ) -> ad.AnnData:
        """Return imputation, cell embeddings, and output gene embeddings."""
        batch_size = self.sc_unveil._positive_integer(batch_size, "batch_size")

        if n_embedding_features is not None:
            n_embedding_features = self.sc_unveil._positive_integer(
                n_embedding_features, "n_embedding_features"
            )
            if n_embedding_features > self.sc_unveil.config.emb_dim:
                raise ValueError(
                    "n_embedding_features cannot exceed the model embedding "
                    f"dimension ({self.sc_unveil.config.emb_dim})."
                )

        if list_of_genes is None:
            enriched = self.get_all_genes_imputation(batch_size=batch_size)
            gene_indices = np.arange(self.sc_unveil.config.n_genes, dtype=np.int64)
        else:
            requested_genes, gene_indices = self.sc_unveil._resolve_requested_genes(
                list_of_genes
            )
            enriched = self.get_specific_genes_imputation(
                list_of_gene_names=requested_genes,
                batch_size=batch_size,
            )

        cell_embeddings = self.get_embeddings(n_features=n_embedding_features)
        gene_embeddings = self.sc_unveil._gene_embedding_array(
            normalize=True,
            indices=gene_indices,
        )

        if cell_embeddings.shape[0] != enriched.n_obs:
            raise RuntimeError(
                "Cell count mismatch between imputation and cell embeddings."
            )
        if gene_embeddings.shape[0] != enriched.n_vars:
            raise RuntimeError(
                "Gene count mismatch between imputation and gene embeddings."
            )

        enriched.obsm["X_scunveil"] = cell_embeddings.copy()
        enriched.varm["scunveil_gene_embeddings"] = gene_embeddings.copy()
        return enriched
