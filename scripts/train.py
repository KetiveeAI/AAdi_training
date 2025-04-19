import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import FiftyOneDataset

# Add the 'models' directory to sys.path
current_dir = os.path.dirname(__file__)
models_dir = os.path.abspath(os.path.join(current_dir, "..", "models"))
sys.path.append(models_dir)

from aadi_rvae import AadiR_VAE

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = FiftyOneDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = AadiR_VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)
recon_loss_fn = nn.MSELoss()

def loss_fn(recon, img, mu, logvar):
    recon_loss = recon_loss_fn(recon, img)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.0001 * kl_div

for epoch in range(100):
    for i, (tokens, images) in enumerate(dataloader):
        input_tokens = {k: v.squeeze(1).to(device) for k, v in tokens.items()}
        images = images.to(device)

        output, mu, logvar = model(input_tokens, images)
        loss = loss_fn(output, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
    print(f"Model saved for epoch {epoch}.")