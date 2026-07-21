import json
import re
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import snapshot_download
from scipy.sparse import csc_matrix, csr_matrix, issparse
from tqdm import tqdm

from ._data_operations import logits_to_CPM, simple_scipy_norm_x
from ._layers import PCAProjection
from ._model import RNABagModel


SCUNVEIL_MODEL_REPO = "thonzik/sc-unveil"
STABLE_VERSION_FILE = "models/stable_version.txt"
NO_INPUT_ANNDATA_TEXT = 'No input AnnData is set. Run "set_input_anndata(...)" first.'

# scUNVEIL covers nearly the complete human reference. Mapping less than half
# of the supplied features or UMI mass almost always indicates a wrong
# identifier column, a non-human input, or incompatible annotation.
MIN_MAPPING_FRACTION = 0.5


@tf.function
def run_tf_model_pred(tf_model, x_input):
    return tf_model(x_input, training=False)


class scUnveil:
    """Inference interface for the pretrained scUNVEIL model.

    Parameters
    ----------
    model_version : str or None, default=None
        Checkpoint version stored in the scUNVEIL Hugging Face repository.
        ``None`` selects the version named in ``models/stable_version.txt``.

    verbose : bool, default=True
        Show model-loading messages and progress bars.
    """

    def __init__(self, model_version=None, verbose=True):
        if not isinstance(verbose, (bool, np.bool_)):
            raise TypeError("verbose must be a boolean.")

        self.verbose = bool(verbose)
        self.input_anndata = None
        self.raw_embeddings = None
        self.pca_embeddings = None
        self.var_map_matrix = None
        self.gene_mapping_summary = None
        self._input_obs = None

        self._message("Model initialization...")

        if model_version is None:
            checkpoint_path = snapshot_download(
                repo_id=SCUNVEIL_MODEL_REPO,
                repo_type="model",
                allow_patterns=[STABLE_VERSION_FILE],
            )
            stable_version_path = Path(checkpoint_path) / STABLE_VERSION_FILE
            try:
                model_version = (
                    stable_version_path.read_text(encoding="utf-8")
                    .splitlines()[0]
                    .strip()
                )
            except (OSError, IndexError) as exc:
                raise RuntimeError(
                    "Could not read the stable scUNVEIL model version from "
                    f"{stable_version_path}."
                ) from exc

        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version must be a non-empty string or None.")

        model_version = model_version.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model_version):
            raise ValueError(
                "model_version may contain only letters, digits, '.', '_', and '-'."
            )
        self.model_version = model_version

        var_sorted_fname = f"models/{model_version}/var_sorted.csv"
        config_fname = f"models/{model_version}/config.json"
        weights_fname = f"models/{model_version}/weights.weights.h5"
        pca_mean_fname = f"models/{model_version}/pca_mean.npy"
        pca_mat_fname = f"models/{model_version}/pca_mat.npy"

        checkpoint_path = snapshot_download(
            repo_id=SCUNVEIL_MODEL_REPO,
            repo_type="model",
            allow_patterns=[
                var_sorted_fname,
                config_fname,
                weights_fname,
                pca_mean_fname,
                pca_mat_fname,
            ],
        )
        self.checkpoint_path = Path(checkpoint_path)

        required_paths = {
            "reference genes": self.checkpoint_path / var_sorted_fname,
            "configuration": self.checkpoint_path / config_fname,
            "weights": self.checkpoint_path / weights_fname,
            "PCA mean": self.checkpoint_path / pca_mean_fname,
            "PCA matrix": self.checkpoint_path / pca_mat_fname,
        }
        missing_files = [
            f"{label}: {path}"
            for label, path in required_paths.items()
            if not path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                "The downloaded checkpoint is incomplete:\n" + "\n".join(missing_files)
            )

        self.reference_var = pd.read_csv(required_paths["reference genes"])

        try:
            with required_paths["configuration"].open(
                "r", encoding="utf-8"
            ) as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("The checkpoint configuration is invalid.") from exc

        self.config = SimpleNamespace(**config)
        self._validate_checkpoint_metadata()
        self._build_reference_lookups()

        model = RNABagModel(
            n_vars=self.config.n_genes,
            n_layers=self.config.n_layers,
            emb_dim=self.config.emb_dim,
        )
        self._message("Loading weights...")
        model.model.load_weights(required_paths["weights"])

        self.full_model = model.model
        self.output_layer = model.model.layers[-1]

        if not isinstance(self.output_layer, tf.keras.layers.Dense):
            raise RuntimeError("The checkpoint model does not end in a Dense layer.")
        if self.output_layer.units != self.config.n_genes:
            raise RuntimeError(
                "The checkpoint output layer does not match config.n_genes."
            )

        embedding_output = model.model.layers[-2].output
        if int(embedding_output.shape[-1]) != self.config.emb_dim:
            raise RuntimeError(
                "The checkpoint embedding layer does not match config.emb_dim."
            )

        self.raw_embedder = tf.keras.Model(
            inputs=model.model.input,
            outputs=embedding_output,
        )
        self.expression_predictor = tf.keras.Sequential([self.output_layer])

        pca_mean = np.load(required_paths["PCA mean"]).astype(np.float32)
        if pca_mean.shape == (self.config.emb_dim,):
            pca_mean = pca_mean[None, :]
        if pca_mean.shape != (1, self.config.emb_dim):
            raise RuntimeError(
                "The checkpoint PCA mean must have shape "
                f"({self.config.emb_dim},) or (1, {self.config.emb_dim})."
            )

        pca_mat = np.load(required_paths["PCA matrix"]).astype(np.float32)
        expected_pca_shape = (self.config.emb_dim, self.config.emb_dim)
        if pca_mat.shape != expected_pca_shape:
            raise RuntimeError(
                "The checkpoint PCA matrix must have shape "
                f"{expected_pca_shape}, received {pca_mat.shape}."
            )
        if not np.isfinite(pca_mean).all() or not np.isfinite(pca_mat).all():
            raise RuntimeError("The checkpoint PCA arrays contain non-finite values.")

        self.pca_mean = pca_mean
        self.pca_mat = pca_mat
        self.pca_projector = tf.keras.Sequential([PCAProjection()])
        self.pca_projector.build((None, self.config.emb_dim))
        self.pca_projector.set_weights([self.pca_mean, self.pca_mat])

    def _message(self, message):
        if self.verbose:
            print(message)

    @staticmethod
    def _positive_integer(value, name):
        if not isinstance(value, (int, np.integer)) or isinstance(
            value, (bool, np.bool_)
        ):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    def _validate_checkpoint_metadata(self):
        required_config = ("n_genes", "n_layers", "emb_dim")
        missing_config = [
            name for name in required_config if not hasattr(self.config, name)
        ]
        if missing_config:
            raise RuntimeError(
                "The checkpoint configuration is missing required fields: "
                f"{missing_config}."
            )

        for name in required_config:
            value = getattr(self.config, name)
            if not isinstance(value, (int, np.integer)) or isinstance(
                value, (bool, np.bool_)
            ):
                raise RuntimeError(f"config.{name} must be an integer.")
            if value <= 0:
                raise RuntimeError(f"config.{name} must be positive.")

        required_columns = {"feature_id", "feature_name"}
        missing_columns = required_columns.difference(self.reference_var.columns)
        if missing_columns:
            raise RuntimeError(
                "The checkpoint reference table is missing columns: "
                f"{sorted(missing_columns)}."
            )
        if len(self.reference_var) < self.config.n_genes:
            raise RuntimeError(
                "The checkpoint reference table contains fewer rows than "
                "config.n_genes."
            )

    @staticmethod
    def _normalize_identifier(identifier, identifier_type):
        if pd.isna(identifier):
            return None
        identifier = str(identifier).strip()
        if not identifier:
            return None
        if identifier_type == "feature_id":
            match = re.fullmatch(r"(ENSG\d+)\.\d+", identifier)
            if match:
                identifier = match.group(1)
        return identifier

    def _build_reference_lookups(self):
        model_var = self.reference_var.iloc[: self.config.n_genes]

        id_lookup = {}
        duplicate_ids = set()
        for index, identifier in enumerate(model_var["feature_id"]):
            identifier = self._normalize_identifier(identifier, "feature_id")
            if identifier is None:
                continue
            previous = id_lookup.get(identifier)
            if previous is not None and previous != index:
                duplicate_ids.add(identifier)
            else:
                id_lookup[identifier] = index

        if duplicate_ids:
            examples = sorted(duplicate_ids)[:10]
            raise RuntimeError(
                "The model reference contains duplicate feature_id values, "
                f"for example: {examples}."
            )

        symbol_positions = {}
        for index, identifier in enumerate(model_var["feature_name"]):
            identifier = self._normalize_identifier(identifier, "feature_name")
            if identifier is not None:
                symbol_positions.setdefault(identifier, []).append(index)

        symbol_lookup = {
            identifier: positions[0]
            for identifier, positions in symbol_positions.items()
            if len(positions) == 1
        }
        ambiguous_symbols = {
            identifier
            for identifier, positions in symbol_positions.items()
            if len(positions) > 1
        }

        identifier_lookup = dict(id_lookup)
        ambiguous_identifiers = set(ambiguous_symbols)
        for identifier, index in symbol_lookup.items():
            previous = identifier_lookup.get(identifier)
            if previous is not None and previous != index:
                ambiguous_identifiers.add(identifier)
            else:
                identifier_lookup[identifier] = index

        self._reference_id_lookup = id_lookup
        self._reference_symbol_lookup = symbol_lookup
        self._ambiguous_reference_symbols = ambiguous_symbols
        self._identifier_lookup = identifier_lookup
        self._ambiguous_output_identifiers = ambiguous_identifiers

    def _validate_anndata_structure(self, input_anndata):
        if not isinstance(input_anndata, ad.AnnData):
            raise TypeError("input_anndata must be an anndata.AnnData object.")
        if input_anndata.n_obs <= 0:
            raise ValueError("input_anndata must contain at least one cell.")
        if input_anndata.n_vars <= 0:
            raise ValueError("input_anndata must contain at least one feature.")
        if input_anndata.X is None:
            raise ValueError("input_anndata.X must contain raw UMI counts.")
        if input_anndata.X.shape != input_anndata.shape:
            raise ValueError("input_anndata.X has a shape inconsistent with AnnData.")

    def _require_input(self):
        if (
            self.input_anndata is None
            or self.raw_embeddings is None
            or self.pca_embeddings is None
        ):
            raise RuntimeError(NO_INPUT_ANNDATA_TEXT)

    def set_input_anndata(self, input_anndata, batch_size=32):
        """Validate and process raw UMI counts from an AnnData object.

        The operation is transactional. If validation or model inference fails,
        a previously processed input and its cached embeddings remain intact.
        """
        batch_size = self._positive_integer(batch_size, "batch_size")
        self._validate_anndata_structure(input_anndata)

        (
            var_map_matrix,
            raw_embeddings,
            pca_embeddings,
            mapping_summary,
        ) = self._process_anndata(input_anndata, batch_size=batch_size)

        # Commit the new state only after every validation and inference batch
        # has completed successfully.
        self.input_anndata = input_anndata
        self._input_obs = input_anndata.obs.copy()
        self.var_map_matrix = var_map_matrix
        self.raw_embeddings = raw_embeddings
        self.pca_embeddings = pca_embeddings
        self.gene_mapping_summary = mapping_summary

    def _process_anndata(self, input_anndata, batch_size):
        var_map_matrix, mapping_summary = self._calculate_gene_sort(input_anndata)
        raw_embeddings, pca_embeddings, mapped_umi_fraction = (
            self._calculate_embeddings(
                input_anndata,
                var_map_matrix,
                batch_size,
            )
        )

        mapping_summary["mapped_umi_fraction"] = mapped_umi_fraction
        if (
            mapped_umi_fraction is not None
            and mapped_umi_fraction < MIN_MAPPING_FRACTION
        ):
            raise ValueError(
                "Only "
                f"{mapped_umi_fraction:.1%} of input UMI counts map to the "
                "scUNVEIL gene vocabulary. Ensure that .X contains raw human "
                "UMI counts and .var contains compatible human gene symbols "
                "or Ensembl IDs."
            )

        self._message(
            "Mapped "
            f"{mapping_summary['mapped_input_features']:,}/"
            f"{mapping_summary['input_features']:,} input features "
            f"({mapping_summary['mapped_feature_fraction']:.1%}) using "
            f"{mapping_summary['identifier_type']} from "
            f"{mapping_summary['identifier_column']!r}."
        )
        if mapped_umi_fraction is not None:
            self._message(f"Preserved {mapped_umi_fraction:.1%} of input UMI counts.")

        return (
            var_map_matrix,
            raw_embeddings,
            pca_embeddings,
            mapping_summary,
        )

    def _candidate_identifiers(self, input_anndata):
        yield "__index__", input_anndata.var.index
        for column in input_anndata.var.columns:
            values = input_anndata.var[column]
            if values.ndim == 1:
                yield str(column), values

    def _detect_gene_column(self, input_anndata):
        """Find the input identifier source with the best full-vocabulary map."""
        best = None

        for column, values in self._candidate_identifiers(input_anndata):
            raw_values = list(values)
            for identifier_type, lookup, ambiguous in (
                ("feature_id", self._reference_id_lookup, set()),
                (
                    "feature_name",
                    self._reference_symbol_lookup,
                    self._ambiguous_reference_symbols,
                ),
            ):
                input_indices = []
                reference_indices = []
                ambiguous_values = set()

                for input_index, value in enumerate(raw_values):
                    identifier = self._normalize_identifier(value, identifier_type)
                    if identifier is None:
                        continue
                    if identifier in ambiguous:
                        ambiguous_values.add(identifier)
                        continue
                    reference_index = lookup.get(identifier)
                    if reference_index is not None:
                        input_indices.append(input_index)
                        reference_indices.append(reference_index)

                unique_reference_count = len(set(reference_indices))
                score = (
                    unique_reference_count,
                    len(input_indices),
                    1 if identifier_type == "feature_id" else 0,
                )
                result = {
                    "column": column,
                    "identifier_type": identifier_type,
                    "input_indices": input_indices,
                    "reference_indices": reference_indices,
                    "ambiguous_values": ambiguous_values,
                    "score": score,
                }
                if best is None or score > best["score"]:
                    best = result

        if best is None or not best["input_indices"]:
            raise ValueError(
                "Could not find compatible human gene identifiers in "
                "input_anndata.var. Put gene symbols (for example TP53 or "
                "CD3D) or Ensembl IDs (for example ENSG00000141510) in the "
                ".var index or a .var column."
            )

        return best

    def _calculate_gene_sort(self, input_anndata):
        self._message("Mapping gene permutation...")
        detection = self._detect_gene_column(input_anndata)

        input_indices = np.asarray(detection["input_indices"], dtype=np.int64)
        reference_indices = np.asarray(detection["reference_indices"], dtype=np.int64)

        unique_reference_indices, counts = np.unique(
            reference_indices, return_counts=True
        )
        duplicated_targets = unique_reference_indices[counts > 1]
        if len(duplicated_targets):
            duplicate_identifiers = []
            reference_set = set(duplicated_targets.tolist())
            selected_values = list(
                input_anndata.var.index
                if detection["column"] == "__index__"
                else input_anndata.var[detection["column"]]
            )
            for input_index, reference_index in zip(input_indices, reference_indices):
                if int(reference_index) in reference_set:
                    duplicate_identifiers.append(str(selected_values[input_index]))

            raise ValueError(
                "Multiple input features map to the same model gene. "
                "Deduplicate the input gene identifiers before inference. "
                f"Examples: {sorted(set(duplicate_identifiers))[:10]}."
            )

        mapped_feature_fraction = len(input_indices) / input_anndata.n_vars
        if mapped_feature_fraction < MIN_MAPPING_FRACTION:
            raise ValueError(
                "Only "
                f"{len(input_indices):,}/{input_anndata.n_vars:,} input "
                f"features ({mapped_feature_fraction:.1%}) map to the "
                "scUNVEIL gene vocabulary. The best source was "
                f"{detection['column']!r} interpreted as "
                f"{detection['identifier_type']}."
            )

        values = np.ones(len(input_indices), dtype=np.float32)
        var_map_matrix = csc_matrix(
            (values, (input_indices, reference_indices)),
            shape=(input_anndata.n_vars, self.config.n_genes),
        )

        mapping_summary = {
            "identifier_column": detection["column"],
            "identifier_type": detection["identifier_type"],
            "input_features": int(input_anndata.n_vars),
            "mapped_input_features": int(len(input_indices)),
            "mapped_model_features": int(len(unique_reference_indices)),
            "unmapped_input_features": int(input_anndata.n_vars - len(input_indices)),
            "mapped_feature_fraction": float(mapped_feature_fraction),
            "ambiguous_reference_identifiers": sorted(detection["ambiguous_values"]),
        }
        return var_map_matrix, mapping_summary

    @staticmethod
    def _validated_count_batch(batch):
        if issparse(batch):
            batch = batch.tocsr(copy=True)
            batch.sum_duplicates()
            values = batch.data
            if batch.dtype.kind in {"O", "S", "U", "c", "b"}:
                raise TypeError("input_anndata.X must contain numeric UMI counts.")
        else:
            try:
                dense = np.asarray(batch)
            except Exception as exc:
                raise TypeError(
                    "input_anndata.X must be a NumPy-like dense matrix or a "
                    "SciPy-compatible sparse matrix."
                ) from exc
            if dense.ndim != 2:
                raise ValueError("Every input expression batch must be 2D.")
            if dense.dtype.kind in {"O", "S", "U", "c", "b"}:
                raise TypeError("input_anndata.X must contain numeric UMI counts.")
            values = dense.ravel()
            batch = csr_matrix(dense)

        if values.size:
            try:
                finite = np.isfinite(values)
            except TypeError as exc:
                raise TypeError(
                    "input_anndata.X must contain numeric UMI counts."
                ) from exc
            if not finite.all():
                raise ValueError("input_anndata.X contains NaN or infinite values.")
            if np.any(values < 0):
                raise ValueError("input_anndata.X contains negative values.")
            if not np.equal(values, np.floor(values)).all():
                raise ValueError(
                    "input_anndata.X must contain raw integer-like UMI counts. "
                    "Normalized or log-transformed expression is not a valid "
                    "scUNVEIL input."
                )

        return batch

    def _calculate_embeddings(
        self,
        input_anndata,
        var_map_matrix,
        batch_size,
    ):
        self._message("Processing cells...")
        n_cells = input_anndata.n_obs
        raw_embeddings = np.zeros((n_cells, self.config.emb_dim), dtype=np.float16)
        pca_embeddings = np.zeros((n_cells, self.config.emb_dim), dtype=np.float16)

        total_umis = 0.0
        mapped_umis = 0.0

        with tqdm(
            total=n_cells,
            disable=not self.verbose,
            desc="scUNVEIL cells",
            unit="cell",
        ) as progress:
            for start in range(0, n_cells, batch_size):
                end = min(start + batch_size, n_cells)
                try:
                    count_batch = input_anndata.X[start:end]
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not read cells {start}:{end} from AnnData.X."
                    ) from exc

                count_batch = self._validated_count_batch(count_batch)
                mapped_batch = count_batch @ var_map_matrix

                total_umis += float(count_batch.sum(dtype=np.float64))
                mapped_umis += float(mapped_batch.sum(dtype=np.float64))

                model_input = simple_scipy_norm_x(mapped_batch)
                batch_size_actual = int(model_input.shape[0])

                raw_batch = run_tf_model_pred(self.raw_embedder, model_input)
                raw_batch_array = raw_batch.numpy()
                if not np.isfinite(raw_batch_array).all():
                    raise FloatingPointError(
                        "The model produced non-finite raw embeddings. Verify "
                        "that the input contains valid raw UMI counts."
                    )
                raw_embeddings[start : start + batch_size_actual] = raw_batch_array

                pca_batch = run_tf_model_pred(self.pca_projector, raw_batch)
                pca_batch_array = pca_batch.numpy()
                if not np.isfinite(pca_batch_array).all():
                    raise FloatingPointError(
                        "The PCA projection produced non-finite embeddings."
                    )
                pca_embeddings[start : start + batch_size_actual] = pca_batch_array
                progress.update(batch_size_actual)

        mapped_umi_fraction = mapped_umis / total_umis if total_umis > 0 else None
        return raw_embeddings, pca_embeddings, mapped_umi_fraction

    def _model_var(self, indices=None):
        model_var = self.reference_var.iloc[: self.config.n_genes]
        if indices is not None:
            model_var = model_var.iloc[np.asarray(indices, dtype=np.int64)]
        model_var = model_var.copy()

        feature_ids = model_var["feature_id"].astype(str)
        if feature_ids.is_unique:
            model_var.index = pd.Index(feature_ids, name=None)
        return model_var

    def get_raw_embeddings(self):
        """Return the unrotated 2,048-dimensional model embeddings."""
        self._require_input()
        return self.raw_embeddings.copy()

    def get_embeddings(self, n_features=512):
        """Return PCA-ordered cell embeddings.

        ``None`` returns all PCA components. A positive integer returns that
        many leading components. Use :meth:`get_raw_embeddings` for the
        unrotated hidden state.
        """
        self._require_input()
        if n_features is None:
            return self.pca_embeddings.copy()

        n_features = self._positive_integer(n_features, "n_features")
        if n_features > self.config.emb_dim:
            raise ValueError(
                "n_features cannot exceed the model embedding dimension "
                f"({self.config.emb_dim})."
            )
        return self.pca_embeddings[:, :n_features].copy()

    def _predict_log10_cpm(self, raw_embeddings):
        prediction = run_tf_model_pred(self.expression_predictor, raw_embeddings)
        prediction = logits_to_CPM(prediction)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("The model produced non-finite log10(CPM) values.")
        return prediction

    def get_all_genes_imputation(self, batch_size=128):
        """Return imputed expression for all model genes as log10(CPM)."""
        self._require_input()
        batch_size = self._positive_integer(batch_size, "batch_size")

        n_cells = self.raw_embeddings.shape[0]
        output_gib = (n_cells * self.config.n_genes * np.dtype(np.float16).itemsize) / (
            1024**3
        )
        if output_gib >= 1:
            self._message(
                "Allocating approximately "
                f"{output_gib:.1f} GiB for all-gene imputation. Use "
                "get_specific_genes_imputation for a smaller result."
            )

        gene_expressions = np.zeros((n_cells, self.config.n_genes), dtype=np.float16)
        with tqdm(
            total=n_cells,
            disable=not self.verbose,
            desc="scUNVEIL imputation",
            unit="cell",
        ) as progress:
            for start in range(0, n_cells, batch_size):
                raw_batch = self.raw_embeddings[start : start + batch_size]
                prediction = self._predict_log10_cpm(raw_batch)
                n_batch = raw_batch.shape[0]
                gene_expressions[start : start + n_batch] = prediction
                progress.update(n_batch)

        return ad.AnnData(
            X=gene_expressions,
            obs=self._input_obs.copy(),
            var=self._model_var(),
        )

    def _resolve_requested_genes(self, list_of_gene_names):
        if isinstance(list_of_gene_names, str):
            requested_genes = [list_of_gene_names]
        else:
            try:
                requested_genes = list(list_of_gene_names)
            except TypeError as exc:
                raise TypeError(
                    "list_of_gene_names must be a string or an iterable of strings."
                ) from exc

        if not requested_genes:
            raise ValueError("At least one gene identifier must be provided.")
        if any(
            not isinstance(gene, str) or not gene.strip() for gene in requested_genes
        ):
            raise ValueError("Every gene identifier must be a non-empty string.")

        requested_genes = [gene.strip() for gene in requested_genes]
        duplicate_identifiers = pd.Index(requested_genes)
        duplicate_identifiers = (
            duplicate_identifiers[duplicate_identifiers.duplicated()].unique().tolist()
        )
        if duplicate_identifiers:
            raise ValueError(
                f"Duplicate gene identifiers were requested: {duplicate_identifiers}."
            )

        gene_indices = []
        missing = []
        ambiguous = []
        for gene in requested_genes:
            normalized_id = self._normalize_identifier(gene, "feature_id")
            if gene in self._ambiguous_output_identifiers:
                ambiguous.append(gene)
                continue
            index = self._identifier_lookup.get(gene)
            if index is None and normalized_id != gene:
                index = self._identifier_lookup.get(normalized_id)
            if index is None:
                missing.append(gene)
            else:
                gene_indices.append(index)

        if ambiguous:
            raise ValueError(
                "The following identifiers map to multiple model genes; use "
                f"their Ensembl feature_id values instead: {ambiguous}."
            )
        if missing:
            raise ValueError(
                "The following genes are not present in the model reference: "
                f"{missing}."
            )

        duplicate_positions = pd.Index(gene_indices)
        duplicate_positions = (
            duplicate_positions[duplicate_positions.duplicated()].unique().tolist()
        )
        if duplicate_positions:
            raise ValueError(
                "Multiple requested identifiers refer to the same model gene. "
                "Request each gene only once."
            )

        return requested_genes, np.asarray(gene_indices, dtype=np.int64)

    def get_specific_genes_imputation(self, list_of_gene_names, batch_size=128):
        """Return selected-gene imputation in request order as log10(CPM)."""
        self._require_input()
        batch_size = self._positive_integer(batch_size, "batch_size")
        _, gene_indices = self._resolve_requested_genes(list_of_gene_names)

        n_cells = self.raw_embeddings.shape[0]
        gene_expressions = np.zeros((n_cells, len(gene_indices)), dtype=np.float16)
        with tqdm(
            total=n_cells,
            disable=not self.verbose,
            desc="scUNVEIL imputation",
            unit="cell",
        ) as progress:
            for start in range(0, n_cells, batch_size):
                raw_batch = self.raw_embeddings[start : start + batch_size]
                prediction = self._predict_log10_cpm(raw_batch)
                selected_prediction = prediction[:, gene_indices]
                n_batch = raw_batch.shape[0]
                gene_expressions[start : start + n_batch] = selected_prediction
                progress.update(n_batch)

        return ad.AnnData(
            X=gene_expressions,
            obs=self._input_obs.copy(),
            var=self._model_var(gene_indices),
        )

    def _gene_embedding_array(self, normalize=True, indices=None):
        if not isinstance(normalize, (bool, np.bool_)):
            raise TypeError("normalize must be a boolean.")

        kernel = tf.cast(self.output_layer.kernel, tf.float32)
        scale = None
        if normalize:
            # Always scale against the complete decoder matrix. A selected
            # subset must equal the same rows from get_genes_embeddings().
            scale = tf.math.reduce_std(kernel)

        if indices is None:
            gene_embeddings = tf.transpose(kernel)
        else:
            indices = tf.convert_to_tensor(indices, dtype=tf.int32)
            gene_embeddings = tf.transpose(tf.gather(kernel, indices, axis=1))

        if normalize:
            gene_embeddings = gene_embeddings / tf.maximum(scale, 1e-9)

        result = gene_embeddings.numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("Gene embeddings contain non-finite values.")
        return result.astype(np.float16)

    def get_genes_embeddings(self, normalize=True):
        """Return output-decoder gene embeddings.

        Rows correspond to model genes and columns to the cell-embedding space.
        With ``normalize=True``, the complete matrix is scaled by its global
        standard deviation, preserving relative vector lengths and cosines.
        """
        gene_embeddings = self._gene_embedding_array(normalize=normalize)
        return ad.AnnData(
            X=gene_embeddings,
            obs=self._model_var(),
        )

    def generate_cells(
        self,
        n_cells,
        sampling_depth,
        batch_size=128,
        seed=None,
    ):
        """Generate count vectors by autoregressive next-UMI sampling."""
        n_cells = self._positive_integer(n_cells, "n_cells")
        sampling_depth = self._positive_integer(sampling_depth, "sampling_depth")
        batch_size = self._positive_integer(batch_size, "batch_size")
        if seed is not None:
            if not isinstance(seed, (int, np.integer)) or isinstance(
                seed, (bool, np.bool_)
            ):
                raise TypeError("seed must be an integer or None.")
            seed = int(seed) % (2**31 - 1)

        cells = tf.zeros((n_cells, self.config.n_genes), dtype=tf.float32)
        random_call = 0

        with tqdm(
            total=n_cells * sampling_depth,
            disable=not self.verbose,
            desc="scUNVEIL generation",
            unit="UMI",
        ) as progress:
            for _ in range(sampling_depth):
                for batch_start in range(0, n_cells, batch_size):
                    batch_end = min(batch_start + batch_size, n_cells)
                    batch_cells = tf.math.log1p(cells[batch_start:batch_end])
                    next_umi_logits = run_tf_model_pred(self.full_model, batch_cells)

                    if seed is None:
                        sampled_indices = tf.random.categorical(
                            next_umi_logits,
                            num_samples=1,
                            dtype=tf.int32,
                        )[:, 0]
                    else:
                        sampled_indices = tf.random.stateless_categorical(
                            next_umi_logits,
                            num_samples=1,
                            seed=tf.constant([seed, random_call], dtype=tf.int32),
                            dtype=tf.int32,
                        )[:, 0]
                        random_call += 1

                    n_samples = tf.shape(sampled_indices)[0]
                    update_indices = tf.stack(
                        [
                            tf.range(
                                batch_start,
                                batch_start + n_samples,
                                dtype=tf.int32,
                            ),
                            sampled_indices,
                        ],
                        axis=1,
                    )
                    cells = tf.tensor_scatter_nd_add(
                        cells,
                        indices=update_indices,
                        updates=tf.ones([n_samples], dtype=cells.dtype),
                    )
                    progress.update(batch_end - batch_start)

        return ad.AnnData(
            X=csr_matrix(cells.numpy()),
            var=self._model_var(),
        )

    def get_fully_enriched_h5ad(
        self,
        batch_size=128,
        list_of_genes=None,
        n_embedding_features=None,
    ):
        """Return imputation, cell embeddings, and output gene embeddings."""
        self._require_input()
        batch_size = self._positive_integer(batch_size, "batch_size")

        if n_embedding_features is not None:
            n_embedding_features = self._positive_integer(
                n_embedding_features, "n_embedding_features"
            )
            if n_embedding_features > self.config.emb_dim:
                raise ValueError(
                    "n_embedding_features cannot exceed the model embedding "
                    f"dimension ({self.config.emb_dim})."
                )

        if list_of_genes is None:
            enriched = self.get_all_genes_imputation(batch_size=batch_size)
            gene_indices = np.arange(self.config.n_genes, dtype=np.int64)
        else:
            _, gene_indices = self._resolve_requested_genes(list_of_genes)
            enriched = self.get_specific_genes_imputation(
                list_of_gene_names=list_of_genes,
                batch_size=batch_size,
            )

        cell_embeddings = self.get_embeddings(n_features=n_embedding_features)
        gene_embeddings = self._gene_embedding_array(
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
