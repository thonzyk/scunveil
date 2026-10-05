# Set up a clean Python 3.11 environment before installing scUNVEIL.
# Install scUNVEIL inside the activated environment:
#   pip install scunveil

import anndata as ad
from scunveil import scUnveil

# Standard loading of an H5AD file (change path to your H5AD here)
adata = ad.read_h5ad("test_shard.h5ad")

# Simple initialization of the scUNVEIL model
sc_unveil = scUnveil()

# Process the input into an independent result tied to this model.
# The H5AD file must contain a valid .var with either Gene Names or Ensembl IDs.
# .X must contain raw, nonnegative, integer-like UMI counts.
result = sc_unveil.process_anndata(adata, batch_size=1_000)

# Get embeddings for the input cells.
# n_features can be set between 1 and 2048.
# The default is 512; None returns all 2048 PCA components.
embeddings = result.get_embeddings(n_features=512)
print("embeddings:\n", embeddings)
del embeddings

# Get embeddings for the input cells with full dimensionality,
# without PCA-based feature sorting.
raw_embeddings = result.get_raw_embeddings()
print("raw_embeddings:\n", raw_embeddings)
del raw_embeddings

# Get log10(CPM) imputation for all genes with shape = (n_cells, 60_000).
# This allocates a large dense matrix; use selected-gene imputation when possible.
imputed_expressions = result.get_all_genes_imputation(batch_size=1_000)
print("imputed_expressions.var:\n", imputed_expressions.var)
print("imputed_expressions.X:\n", imputed_expressions.X)
del imputed_expressions

# Get log10(CPM) imputation for selected genes.
imputed_genes = result.get_specific_genes_imputation(
    ["CD4", "CD8A"], batch_size=1000
)
print("imputed_genes.var:\n", imputed_genes.var)
print("imputed_genes.X:\n", imputed_genes.X)
del imputed_genes

# Get shared input/output gene embeddings.
gene_embeddings = sc_unveil.get_genes_embeddings()
print("gene_embeddings.obs:\n", gene_embeddings.obs)
print("gene_embeddings.X:\n", gene_embeddings.X)
del gene_embeddings

# Generate artificial cells
generated_cells = sc_unveil.generate_cells(
    n_cells=128,
    sampling_depth=1024,
    batch_size=128,
    seed=7,
)
print("generated_cells.X:\n", generated_cells.X)
del generated_cells

# Get fully enriched h5ad (embeddings + imputation)
enriched_h5ad = result.get_fully_enriched_h5ad(batch_size=1000)
print("enriched_h5ad.var:\n", enriched_h5ad.var)
print("enriched_h5ad.X:\n", enriched_h5ad.X)
del enriched_h5ad
