# EEG-Hyena

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/eeg-hyena.svg)](https://badge.fury.io/py/eeg-hyena)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()

Adapted Hyena Hierarchy for real-time EEG signal-to-text generation using PyTorch and MNE for biosignal processing. This project extends the original Hyena convolutional language model to handle EEG inputs, preprocessing signals into sequences for character-level prediction.

## Acknowledgments
This project is based on the [Hyena-Hierarchy](https://github.com/Suro-One/Hyena-Hierarchy) by Suro-One (@OASIS_Suro_One). The core model architecture and EWC implementation are adapted from their work on convolutional LMs inspired by the Hyena Hierarchy paper.

## Installation
```bash
pip install -e .
```

## Usage
```python
import mne
from eeg_hyena import EEGHyenaModel, preprocess_eeg

# Load sample EEG data (e.g., from MNE sample dataset)
raw = mne.io.read_raw_edf('path/to/eeg.edf', preload=True)

# Preprocess
features = preprocess_eeg(raw)

# Initialize model
model = EEGHyenaModel(vocab_size=256, d_model=512, n_layers=6)  # ASCII chars

# Train or infer...
```

## Features
- EEG preprocessing with filtering, feature extraction.
- Hyena-based sequence modeling for text generation.
- EWC for continual learning.
- Production-ready with packaging.

## Contributing
Fork and PR!

## License
MIT
