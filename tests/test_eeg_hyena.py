import pytest
import numpy as np
import torch
import mne
from eeg_hyena import EEGHyenaModel, preprocess_eeg

def test_preprocess_eeg():
    # Synthetic EEG
    info = mne.create_info(ch_names=['EEG1'], sfreq=250, ch_types=['eeg'])
    data = np.random.randn(1, 1000)
    raw = mne.io.RawArray(data, info)
    features = preprocess_eeg(raw)
    assert features.shape[2] == 64  # n_components

def test_model_forward():
    model = EEGHyenaModel(vocab_size=256, d_model=512, n_layers=1, feature_dim=64)  # Small for test
    inputs = torch.randn(1, 100, 64)
    outputs = model(inputs)
    assert outputs.shape == (1, 100, 256)

def test_training():
    from train import generate_synthetic_data
    features, labels = generate_synthetic_data(n_samples=1, seq_len=100, n_channels=1)
    assert features.shape == (1, 100, 64)
    assert labels.shape == (1, 100)
