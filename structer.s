AADI_Project/
│
├── data/                   # Raw and processed datasets
│   ├── laion/              # LAION-400M subset
│   │   ├── images/         # Downloaded images
│   │   └── metadata.csv    # Text-image pairs
│   ├── coco/               # COCO dataset
│   ├── textures/           # 3D texture datasets (Polyhaven, TextureLib)
│   │   ├── pbr/            # Albedo, roughness, normal maps
│   │   └── metadata.json   # Texture descriptions
│   └── unlabeled/          # Unlabeled 2017 images
│       ├── images/
│       └── synthetic_captions.csv  # BLIP-generated captions
│
├── src/                    # Training/generation code
│   ├── data_processing/
│   │   ├── resize_images.py        # Resize to 512x512
│   │   ├── caption_generation.py   # BLIP synthetic captions
│   │   └── filter_artifacts.py     # Remove flawed data
│   │
│   ├── models/
│   │   ├── diffusion_unet.py       # Custom U-Net for 3D textures
│   │   ├── vq_vae.py              # VQ-VAE-2 for discrete encoding
│   │   └── clip_text_encoder.py    # Text conditioning
│   │
│   ├── training/
│   │   ├── train_diffusion.py      # Main training script
│   │   ├── losses.py               # LPIPS, edge-aware losses
│   │   └── multi_gpu_train.sh      # Distributed training
│   │
│   └── inference/
│       ├── generate_images.py      # Text-to-image
│       ├── texture_tools/          # 3D validation
│       │   ├── tile_check.py       # Seamless tiling
│       │   └── pbr_validator.py   # Physically-based rendering checks
│       └── app.py                 # Gradio/FastAPI demo
│
├── configs/                # Hyperparameters
│   ├── base.yaml           # Default training configs
│   └── texture.yaml       # 3D-specific settings
│
├── outputs/                # Generated assets
│   ├── images/             # Final renders
│   └── textures/           # PBR texture sets
│
├── requirements.txt        # Python dependencies
├── README.md               # Project setup guide
└── LICENSE