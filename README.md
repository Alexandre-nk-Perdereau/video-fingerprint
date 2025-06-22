# Video Fingerprint

A deep learning model for video fingerprinting and duplicate detection using temporal attention mechanisms.
This project is a draft.

## Features

- **Temporal Attention**: Uses multi-head attention to capture temporal relationships between frames
- **Variable Length Support**: Handles videos of different lengths through adaptive pooling
- **Contrastive Learning**: Learns robust embeddings using contrastive loss with segment augmentation
- **Duplicate Detection**: Efficiently finds duplicate videos based on fingerprint similarity
- **Scalable**: Supports FAISS for fast similarity search on large datasets

## Installation

```bash
uv sync
```

## Data

Download [UCF-101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) data and put the content in data (or in any folder)

## Training

Train the model on your video dataset:

```bash
uv run train.py --data_dir data --batch_size 8 --epoch 100 --num_workers 8
```

Options:
- `--data_dir`: Path to video dataset
- `--batch_size`: Training batch size (default: 8)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--num_workers`: Number of data loading workers (default: 4)
- `--checkpoint`: Resume training from checkpoint
- `--no_amp`: Disable mixed precision training

## Duplicate Detection

Scan a directory for duplicate videos:

```bash
uv run fingerprint.py --model .\checkpoints\best.pth --scan path/to/videos --threshold 0.95
```

Options:
- `--model`: Path to trained model checkpoint
- `--scan`: Directory containing videos to scan
- `--threshold`: Similarity threshold for duplicates (0-1, default: 0.95)
- `--output`: JSON file to save results
- `--device`: Device to use (cuda/cpu, default: cuda)
- `--extensions`: Video file extensions to scan (default: .mp4 .avi .mov .mkv)

## Model Architecture

The model consists of:

1. **Spatial Encoder**: Lightweight CNN to extract frame features (64x64 -> 128d)
2. **Temporal Encoding**: Positional encoding + multi-scale temporal convolutions
3. **Attention Blocks**: Multiple self-attention layers for temporal modeling
4. **Adaptive Pooling**: Combines average, max, and weighted pooling
5. **Final Projection**: Projects to fixed-size embedding (default: 256d)

## Dataset

The dataset loader supports:
- Variable length videos (automatically subsampled if too long)
- Data augmentation (color jittering, flipping, noise, compression)
- Contrastive pair generation with controlled overlap
- Efficient frame loading with PyAV

## Results

The model outputs normalized embeddings that can be used for:
- Duplicate detection (cosine similarity)
- Video retrieval
- Clustering similar content

## License

MIT