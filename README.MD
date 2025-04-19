# AadiAI R-VAE

AadiAI R-VAE (Reconstruction Variational Autoencoder) is a hybrid architecture that fuses text and image representations to generate high-resolution images from a shared latent space. This architecture enables flexible training for applications in 3D texture generation, style transfer, creative generation, and more.

![Architecture](aadi\ r-vae.png)

---

## 📁 Project Structure

```
AadiAI-RVAE/
├── checkpoints/                 # Saved model weights
├── configs/
│   └── train_config.yaml        # Training hyperparameters and settings
├── data/
│   ├── coco/                    # Example dataset folder
│   ├── lalion/                  # Custom dataset folder
│   ├── raw/                     # Raw unprocessed data
│   └── annotations.jsonl        # Annotation file
├── logs/                        # Training and validation logs
├── models/
│   ├── aadi_rvae.py             # Main R-VAE architecture
│   ├── text_encoder.py          # Text encoder model
│   └── unet.py                  # Optional image decoder/UNet model
├── scripts/
│   ├── dataloader.py            # Data loading scripts
│   ├── train.py                 # Training script
│   ├── eval.py                  # Evaluation script
│   ├── rvae_handler.py          # Utility for encoding/decoding
│   └── sample.py                # Sampling from the latent space
├── test/
│   ├── test_r-vae.py            # Test cases
│   ├── requirements.txt         # Dependencies
├── utils/
│   ├── logger.py                # Logging utilities
│   ├── metrics.py               # Metrics for evaluation
│   └── scheduler.py             # Learning rate scheduler
├── venv/                        # Python virtual environment (recommended local only)
├── .gitignore
├── .gitattributes
└── README.md                    # You are here
```

---

## 🧪 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ketivee-org/AadiAI-RVAE.git
cd AadiAI-RVAE
```

### 2. Create and Activate Virtual Environment
```bash
# For Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# For Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r test/requirements.txt
```

---

## 🚀 Quick Start

### Train the model:
```bash
python scripts/train.py --config configs/train_config.yaml
```

### Sample from latent space:
```bash
python scripts/sample.py
```

### Run evaluation:
```bash
python scripts/eval.py
```

---

## 🌐 Links

- 🔗 **Main Website**: [ketivee.com](https://ketivee.com)  
- 📚 **Documentation**: [docs.ketivee.com](https://docs.ketivee.com)  
- ❤️ **Support & Donate**: [donation.ketivee.com](https://donation.ketivee.com)  
- 💌 **Contact**: [support@ketivee.com](mailto:support@ketivee.com)  
- 🧪 **Interface**: _Under Development_

---

## 🧠 Contribution

We welcome community contributions! Feel free to open issues, submit pull requests, or suggest features on the [GitHub page](https://github.com/ketivee-org/AadiAI-RVAE).

---

## 📜 License

This project is under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

> Built with 💡 by [Ketivee](https://ketivee.com)
