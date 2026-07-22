<p align="center"> <img src="https://raw.githubusercontent.com/thonzyk/scunveil/main/assets/scunveil-title.png" alt="scUNVEIL" height="120"> </p>

[![PyPI version](https://img.shields.io/pypi/v/scunveil)](https://pypi.org/project/scunveil/)
![Python versions](https://img.shields.io/pypi/pyversions/scunveil)
[![License](https://img.shields.io/pypi/l/scunveil)](LICENSE)

`scunveil` is the inference package for scUNVEIL, a pretrained model for
single-cell RNA-seq measurement enrichment. It produces reusable cell
embeddings and predicts a depth-enriched transcriptome composition from raw UMI
counts.

The model operates on one cell-level count vector at a time. It does not require
cell-type, tissue, donor, batch, or dataset labels, and it does not perform
dataset-specific training during inference.

## Installation

scUNVEIL supports Python 3.9 through 3.11.

```bash
python -m pip install scunveil
```

On supported Linux x86-64 systems, TensorFlow's CUDA dependencies can be
requested with:

```bash
python -m pip install "scunveil[cuda]"
```

The first `scUnveil()` initialization downloads the selected pretrained
checkpoint from the
[scUNVEIL model repository](https://huggingface.co/thonzik/sc-unveil). Later
initializations reuse the Hugging Face cache.

## Input requirements

`set_input_anndata` accepts an `anndata.AnnData` containing raw, nonnegative,
integer-like UMI counts in `.X`. Dense NumPy arrays, SciPy sparse matrices, and
backed sparse H5AD matrices are supported.

Do not pass normalized or log-transformed expression. If raw counts are stored
in a layer, select them explicitly before inference:

```python
adata_for_scunveil = adata.copy()
adata_for_scunveil.X = adata.layers["counts"].copy()
```

The `.var` index or one of its columns must contain human gene symbols or
Ensembl gene IDs. Versioned Ensembl IDs such as `ENSG00000141510.18` are
accepted. The package compares all candidate identifier columns with the full
model vocabulary, preferring Ensembl IDs when mappings are equivalent.

At least half of the input features and half of the nonzero UMI mass must map to
the model vocabulary. Ambiguous symbols are not assigned arbitrarily; use their
Ensembl IDs instead. Mapping information is available after processing:

```python
print(model.gene_mapping_summary)
```

## Basic usage

```python
import anndata as ad
from scunveil import scUnveil

adata = ad.read_h5ad("my_raw_counts.h5ad")

model = scUnveil()
model.set_input_anndata(adata, batch_size=256)

# Leading PCA-ordered components, shape: (cells, 512)
embeddings = model.get_embeddings(n_features=512)

# Selected-gene depth-enriched expression
markers = model.get_specific_genes_imputation(
    ["CD4", "CD8A", "ENSG00000163599"],
    batch_size=256,
)
```

Pass `verbose=False` to suppress package messages and progress bars:

```python
model = scUnveil(verbose=False)
```

An explicit checkpoint can be selected with
`scUnveil(model_version="VERSION")`. The default follows
`models/stable_version.txt` in the model repository.

## Cell embeddings

```python
# Default: the leading 512 PCA components
x_512 = model.get_embeddings()

# All 2048 PCA components
x_pca_full = model.get_embeddings(n_features=None)

# Original, unrotated 2048-dimensional hidden state
x_raw = model.get_raw_embeddings()
```

The PCA projection was fitted after pretraining so that the leading columns are
ordered by explained variation. A full PCA rotation preserves the complete
embedding space; truncation selects its leading components.

## Expression imputation

```python
selected = model.get_specific_genes_imputation(["CD4", "CD8A"])
all_genes = model.get_all_genes_imputation()
```

Both methods return an `AnnData`. Its `.X` contains:

```text
log10(predicted gene probability) + 6 = log10(CPM)
```

These values are **log10 counts per million**, not raw counts, probabilities, or
ordinary CPM. Selected-gene predictions are normalized against the complete
60,000-gene output vocabulary and preserve request order.

All-gene imputation allocates a dense `(n_cells, 60_000)` float16 matrix. Its
approximate output size is:

```text
n_cells × 60,000 × 2 bytes
```

Use selected-gene imputation when the complete matrix is unnecessary.

## Gene embeddings

```python
gene_embeddings = model.get_genes_embeddings(normalize=True)
```

This returns an `AnnData` with genes in `.obs` and decoder embeddings in `.X`.
For output gene `g`, its vector is the corresponding column of the final output
kernel, transposed into `(n_genes, embedding_dimension)` form. It describes how
directions in cell-embedding space change that gene's predicted logit.

With `normalize=True`, the complete matrix is divided by its global standard
deviation. Relative vector lengths and cosine relationships are preserved. The
decoder bias is not included in the embedding vectors.

## Autoregressive cell generation

```python
generated = model.generate_cells(
    n_cells=128,
    sampling_depth=1_024,
    batch_size=128,
    seed=7,
)
```

Generation starts with empty cells and repeatedly samples the next UMI from the
model's categorical prediction before adding it to the count vector. Every
returned row contains exactly `sampling_depth` UMIs.

## Enriched AnnData

```python
enriched = model.get_fully_enriched_h5ad(
    batch_size=256,
    list_of_genes=["CD4", "CD8A"],
    n_embedding_features=512,
)
```

The result contains:

- Imputed log10(CPM) in `.X`.
- PCA cell embeddings in `.obsm["X_scunveil"]`.
- Output-decoder gene embeddings in
  `.varm["scunveil_gene_embeddings"]`.

Omit `list_of_genes` to include all model genes. Omit
`n_embedding_features` to include all PCA components.

## License

scUNVEIL is distributed under the Apache License 2.0.
