import sys
import os
import pytest
import numpy as np
import torch
import mne
from eeg_hyena import EEGHyenaModel, preprocess_eeg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_preprocess_eeg():
    info = mne.create_info(ch_names=[f"EEG{i}" for i in range(65)], sfreq=250, ch_types=["eeg"] * 65)
    data = np.random.randn(65, 1000)
    raw = mne.io.RawArray(data, info)
    features = preprocess_eeg(raw)
    assert features.shape[2] == 64


def test_model_forward():
    model = EEGHyenaModel(vocab_size=256, d_model=512, n_layers=1, feature_dim=64)
    inputs = torch.randn(1, 100, 64)
    outputs = model(inputs)
    assert outputs.shape == (1, 100, 256)


def test_training():
    from train import generate_synthetic_data

    features, labels = generate_synthetic_data(n_samples=1, seq_len=250, n_channels=65)
    assert features.shape[-1] == 64
    assert labels.shape == (1, 250)


@pytest.mark.skip(reason="Requires MNE sample data download")
def test_mne_sample():
    import mne

    sample_data_path = mne.datasets.sample.data_path()
    raw_path = sample_data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    raw.pick_types(eeg=True)
    features = preprocess_eeg(raw)
    assert len(features) > 0
