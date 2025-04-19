import sys
import os
import torch

# Add the 'models' directory to sys.path
current_dir = os.path.dirname(__file__)
models_dir = os.path.abspath(os.path.join(current_dir, "..", "models"))
sys.path.append(models_dir)

from aadi_rvae import AadiR_VAE
from transformers import BertTokenizer

# Tokenize sample text
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dummy_text = "A beautiful scene with a mountain and lake."
text_tokens = tokenizer(dummy_text, return_tensors="pt")

# Create dummy image and move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_image = torch.randn(1, 3, 1024, 1024)

model = AadiR_VAE().to(device)
text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
dummy_image = dummy_image.to(device)

# Forward pass
output_image, mu, logvar = model(text_tokens, dummy_image)

# Output result
print("Generated Image Shape:", output_image.shape)
print("Latent Mean Shape:", mu.shape)
print("Latent LogVar Shape:", logvar.shape)
print("Model successfully loaded and executed without errors.")
# The output shapes should be consistent with the model's architecture.