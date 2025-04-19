import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, text_tokens):
        return self.bert(**text_tokens).last_hidden_state  # (B, T, 768)


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        x = self.backbone(x)  # (B, 2048, H, W)
        return self.flatten(x)  # (B, 2048, H*W)


class FusionLayer(nn.Module):
    def __init__(self, text_dim=768, img_dim=2048, out_dim=1024):
        super().__init__()
        self.fuse = nn.Linear(text_dim + img_dim, out_dim)

    def forward(self, text_feat, img_feat):
        text_mean = text_feat.mean(dim=1)  # (B, 768)
        img_mean = img_feat.mean(dim=2)    # (B, 2048)
        combined = torch.cat([text_mean, img_mean], dim=1)  # (B, 2816)
        return self.fuse(combined)  # (B, 1024)


class AadiR_VAE(nn.Module):
    def __init__(self, latent_dim=512, target_resolution=(1024, 1024)):
        super().__init__()
        self.text_enc = TextEncoder()
        self.img_enc = ImageEncoder()
        self.fusion = FusionLayer(text_dim=768, img_dim=2048, out_dim=1024)

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        self.target_resolution = target_resolution
        self.decoder = self._build_decoder(latent_dim, target_resolution)

    def _build_decoder(self, latent_dim, target_resolution):
        layers = [
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),
        ]

        current_size = 8
        out_channels = 512
        while current_size < target_resolution[0] or current_size < target_resolution[1]:
            in_channels = out_channels
            out_channels = max(out_channels // 2, 32)  # Reduce channels progressively
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ])
            current_size *= 2

        layers.append(nn.Conv2d(out_channels, 3, kernel_size=3, padding=1))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, text_tokens, image):
        text_feat = self.text_enc(text_tokens)     # (B, T, 768)
        img_feat = self.img_enc(image)             # (B, 2048, H*W)
        fusion = self.fusion(text_feat, img_feat)  # (B, 1024)
        mu, logvar = self.fc_mu(fusion), self.fc_logvar(fusion)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)                      # (B, 3, H, W)
        return out, mu, logvar
