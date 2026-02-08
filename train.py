import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import mne
from eeg_hyena import EEGHyenaModel, preprocess_eeg

# Synthetic EEG and labels (for demo; replace with real labeled data)
def generate_synthetic_data(n_samples=10, seq_len=1000, n_channels=2, vocab_size=256):
    info = mne.create_info(ch_names=[f'EEG{i}' for i in range(n_channels)], sfreq=250, ch_types=['eeg'] * n_channels)
    data = np.random.randn(n_channels, seq_len * n_samples)
    raw = mne.io.RawArray(data, info)
    features = preprocess_eeg(raw)  # (n_samples, seq_len, 64)
    labels = np.random.randint(0, vocab_size, (n_samples, seq_len))  # Random "text" labels
    return features, labels

# Training loop
model = EEGHyenaModel(vocab_size=256, d_model=512, n_layers=6, feature_dim=64)
optimizer = optim.Adam(model.parameters(), lr=0.001)
features, labels = generate_synthetic_data()

for epoch in range(5):  # Short demo
    model.train()
    inputs = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    outputs = model(inputs)
    loss = F.cross_entropy(outputs.view(-1, 256), targets.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), 'model.pth')
