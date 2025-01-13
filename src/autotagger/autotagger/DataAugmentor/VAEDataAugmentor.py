import torch
import torch.nn
import torch.optim
try:
    from .interface import DataAugmentorInterface
    from ..exceptions import ModelNotTrainedError
except:
    class DataAugmentorInterface(object):
        def __init__(self):
            pass
    
    ModelNotTrainedError = RuntimeError

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], latent_dim: int, activate_function: nn.Module):
        super(Encoder, self).__init__()
        layers = []
        in_features = input_dim
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activate_function())
            in_features = out_features
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.model(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_layers: List[int], output_dim: int, activate_function: nn.Module):
        super(Decoder, self).__init__()
        layers = []
        in_features = latent_dim
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activate_function())
            in_features = out_features
        layers.append(nn.Linear(in_features, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class VAEDataAugmentor:
    def __init__(self,
                 data: torch.Tensor,
                 latent_dim: int = 128,
                 encoder_hidden_layers: List[int] = [16, 32],
                 decoder_hidden_layers: List[int] = [32, 16],
                 activate_function: nn.Module = nn.ReLU,
                 criterion: nn.Module = nn.MSELoss
                 ):
        self.latent_dim = latent_dim
        self.criterion = criterion()
        self.encoder = Encoder(data.shape[1], encoder_hidden_layers, latent_dim, activate_function)
        self.decoder = Decoder(latent_dim, decoder_hidden_layers, data.shape[1], activate_function)
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)
        self.data = data

    def fit(self, epochs: int = 100, batch_size: int = 64) -> 'VAEDataAugmentor':
        self.data = self.data.float()
        for epoch in range(epochs):
            for i in range(0, len(self.data), batch_size):
                batch_data = self.data[i:i + batch_size]
                batch_size_current = batch_data.shape[0]

                mu, logvar = self.encoder(batch_data)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                reconstructed_data = self.decoder(z)

                bce_loss = self.criterion(reconstructed_data, batch_data)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = bce_loss + kld_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, num_samples: int) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            generated_data = self.decoder(z)
        return generated_data

    def fit_predict(self, data: torch.Tensor, epochs: int = 100, batch_size: int = 64, num_samples: int = 1000) -> torch.Tensor:
        self.data = data
        self.fit(epochs, batch_size)
        return self.predict(num_samples)

    def __str__(self):
        return f"VAEDataAugmentor(latent_dim={self.latent_dim})"


if __name__ == "__main__":
    
    data = []
    vae_augmentor = VAEDataAugmentor(data, latent_dim=64, encoder_hidden_layers=[256, 128], decoder_hidden_layers=[128, 256])
    vae_augmentor.fit(epochs=50, batch_size=64)
    new_data = vae_augmentor.predict(num_samples=1000)
    print(new_data.shape)  # 应该输出 (1000, num_features)