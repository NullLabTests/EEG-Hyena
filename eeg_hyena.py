import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
import numpy as np
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGHyenaModel(nn.Module):
    """
    Adapted HyenaWithEWC for EEG inputs. Processes EEG features as sequences.
    """
    def __init__(self, vocab_size, d_model, n_layers, dim_feedforward=2048, dropout=0.1, feature_dim=64):
        super(EEGHyenaModel, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)  # Project EEG features to model dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # Hyena layers (adapted from original)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'conv': nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=4,
                    stride=1,
                    padding='same',
                    bias=True,
                    padding_mode='reflect'
                ),
                'gate': nn.Linear(d_model, d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, d_model),
                ),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(n_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)

        # EWC attributes (from original)
        self.old_params = None
        self.fisher_diagonal = None

    def forward(self, src):
        # src: (B, S, feature_dim) -> embed to (B, S, d_model)
        src = self.embedding(src)
        for layer in self.layers:
            x = layer['conv'](src.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(layer['gate'](x))
            x = x * gate
            x = layer['ffn'](x)
            x = layer['dropout'](x)
            src = x
        return self.output(src)

    # EWC methods (adapted from original)
    def calculate_fisher(self, dataset, samples=2000):
        self.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters()}
        for _ in range(samples):
            data = dataset[np.random.randint(len(dataset))].unsqueeze(0).to(device)
            self.zero_grad()
            output = self(data[:, :-1])
            loss = F.cross_entropy(output.view(-1, self.vocab_size), data[:, 1:].view(-1).long())
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += (p.grad.data ** 2) / samples
        self.fisher_diagonal = fisher
        self.old_params = copy.deepcopy(self.state_dict())

    def ewc_loss(self, lamda=15):
        if self.fisher_diagonal is None or self.old_params is None:
            return 0.0
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self.fisher_diagonal:
                loss += (self.fisher_diagonal[n] * (p - self.old_params[n]) ** 2).sum()
        return lamda * loss

def preprocess_eeg(raw, sfreq=250, low_freq=1, high_freq=40, n_components=64):
    """
    Preprocess raw MNE EEG data: filter, epoch, PCA to features.
    Assumes labeled epochs for text prediction (adapt as needed).
    Returns: np.array of shape (samples, seq_len, feature_dim)
    """
    raw.filter(low_freq, high_freq)
    # Example: create fake epochs for demo (replace with real events/labels)
    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(raw, events, preload=True)
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    # Reshape and apply PCA per epoch
    pca = PCA(n_components=n_components)
    features = []
    for ep in data:
        ep_flat = ep.T  # (n_times, n_channels)
        feat = pca.fit_transform(ep_flat)  # (n_times, n_components)
        features.append(feat)
    return np.array(features)  # Adapt to sequence for model

# Example usage (commented)
# model = EEGHyenaModel(vocab_size=256, d_model=512, n_layers=6, feature_dim=64)
# features = preprocess_eeg(raw)  # features: (batch, seq_len, 64)
# outputs = model(torch.tensor(features, dtype=torch.float32))
