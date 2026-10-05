import json

import numpy as np
import pandas as pd
import pytest

import scunveil._inference as inference
from scunveil._model import RNABagModel


@pytest.fixture
def checkpoint(tmp_path):
    version = inference.DEFAULT_MODEL_VERSION
    folder = tmp_path / "models" / version
    folder.mkdir(parents=True)

    config = {"n_genes": 8, "n_layers": 1, "emb_dim": 4, "ff_dim": 6}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame(
        {
            "feature_id": [f"ENSG{i:011d}" for i in range(8)],
            "feature_name": [f"GENE{i}" for i in range(8)],
        }
    ).to_csv(folder / "var_sorted.csv", index=False)
    RNABagModel(8, 1, 4, ff_dim=6).model.save_weights(folder / "weights.weights.h5")
    np.save(folder / "pca_mean.npy", np.zeros(4, dtype=np.float32))
    np.save(folder / "pca_mat.npy", np.eye(4, dtype=np.float32))
    return tmp_path, folder


def test_default_version_downloads_and_loads_expected_files(checkpoint, monkeypatch):
    root, folder = checkpoint
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(root)

    monkeypatch.setattr(inference, "snapshot_download", snapshot_download)
    model = inference.scUnveil(verbose=False)

    assert model.model_version == inference.DEFAULT_MODEL_VERSION
    assert model.config.ff_dim == 6
    assert model.checkpoint_path == root
    assert model.full_model.output_shape == (None, 8)
    assert calls == [
        {
            "repo_id": inference.SCUNVEIL_MODEL_REPO,
            "repo_type": "model",
            "allow_patterns": [
                f"models/{model.model_version}/{name}"
                for name in (
                    "var_sorted.csv",
                    "config.json",
                    "weights.weights.h5",
                    "pca_mean.npy",
                    "pca_mat.npy",
                )
            ],
        }
    ]

    (folder / "pca_mat.npy").unlink()
    with pytest.raises(FileNotFoundError, match="PCA matrix"):
        inference.scUnveil(verbose=False)
