# Set up a clean Python 3.11 environment before installing scUNVEIL.
# Install scUNVEIL inside the activated environment:
#   pip install scunveil

import anndata as ad
from scunveil import scUnveil

# Standard loading of an H5AD file (change path to your H5AD here)
adata = ad.read_h5ad("test_shard.h5ad")

# Simple initialization of the scUNVEIL model
sc_unveil = scUnveil()

# Before running anything else, you must set an input H5AD file.
# The data will be processed automatically and made available for later interaction.
# The H5AD file must contain a valid .var with either Gene Names or Ensembl IDs.
# .X must contain raw, nonnegative, integer-like UMI counts.
sc_unveil.set_input_anndata(adata, batch_size=1_000)

# Get embeddings for the input cells.
# n_features can be set between 1 and 2048.
# The default is 512; None returns all 2048 PCA components.
embeddings = sc_unveil.get_embeddings(n_features=512)
print("embeddings:\n", embeddings)
del embeddings

# Get log10(CPM) imputation for selected genes.
imputed_genes = sc_unveil.get_specific_genes_imputation(
    ["CD4", "CD8A"], batch_size=1000
)
print("imputed_genes.var:\n", imputed_genes.var)
print("imputed_genes.X:\n", imputed_genes.X)
del imputed_genes

