# dataloader_fiftyone.py
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os


class FiftyOneDataset(Dataset):
    def __init__(self, split="quickstart", dataset_dir="data/images"):
        """
        Initialize the FiftyOne dataset.
        Args:
            split (str): Dataset split to use. If "quickstart", downloads a small test dataset.
            dataset_dir (str): Directory to store/load the dataset
        """
        try:
            # Try to load existing dataset
            self.dataset = fo.load_dataset(split)
        except ValueError:
            # If dataset doesn't exist, download quickstart dataset
            if split == "quickstart":
                self.dataset = foz.load_zoo_dataset(
                    "quickstart",
                    dataset_dir=dataset_dir,
                )
            else:
                raise ValueError(f"Dataset '{split}' not found. Please create or import it first.")

        # Ensure the dataset has samples
        if len(self.dataset) == 0:
            raise RuntimeError("Dataset is empty")

        self.samples = self.dataset.select_fields(
            ["filepath", "ground_truth.label"]
        ).values()
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")
            
        image = Image.open(filepath).convert("RGB")
        image = self.transform(image)

        # Use label name or generate dummy text
        text = label if label else "a sample image"
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )

        # Remove the batch dimension added by the tokenizer
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}

        return tokens, image
