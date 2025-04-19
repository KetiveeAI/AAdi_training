import sys
import os

import torch

# Add the parent folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from aadi_rvae import AadiR_VAE

model = AadiR_VAE().cuda()

def generate_image(text: str, image_tensor):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer(text, return_tensors="pt").to("cuda")
    image_tensor = image_tensor.cuda()
    with torch.no_grad():
        output, _, _ = model(tokens, image_tensor)
    return output
