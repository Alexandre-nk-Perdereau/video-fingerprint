import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal attention."""

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SpatialEncoder(nn.Module):
    """Lightweight spatial encoder to reduce 64x64 -> compact features."""

    def __init__(self, in_channels=3, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class TemporalAttentionBlock(nn.Module):
    """Attention block for temporal features with 1D convolutions."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        self.conv1 = nn.Conv1d(dim, dim * mlp_ratio, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(dim * mlp_ratio, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop(attn_out)

        x_norm = self.norm2(x)
        x_conv = rearrange(x_norm, "b t c -> b c t")
        x_conv = self.conv2(self.drop(self.act(self.conv1(x_conv))))
        x_conv = rearrange(x_conv, "b c t -> b t c")
        x = x + self.drop(x_conv)

        return x


class TemporalConvBlock(nn.Module):
    """Temporal convolution block with different scales."""

    def __init__(self, dim, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        dim,
                        dim // len(kernel_sizes),
                        k,
                        padding=k // 2,
                        groups=dim // len(kernel_sizes),
                    ),
                    nn.BatchNorm1d(dim // len(kernel_sizes)),
                    nn.ReLU(inplace=True),
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)


class VideoFingerprintAttention(nn.Module):
    """Fingerprint model for video with attention on all frames."""

    def __init__(
        self,
        spatial_dim=128,
        temporal_dim=256,
        embedding_dim=256,
        num_attention_blocks=4,
        num_heads=8,
    ):
        super().__init__()

        self.spatial_encoder = SpatialEncoder(out_dim=spatial_dim)

        self.temporal_projection = nn.Linear(spatial_dim, temporal_dim)

        self.pos_encoding = PositionalEncoding(temporal_dim)

        self.temporal_conv_blocks = nn.ModuleList(
            [
                TemporalConvBlock(temporal_dim, kernel_sizes=[3, 5, 7, 11])
                for _ in range(2)
            ]
        )

        self.attention_blocks = nn.ModuleList(
            [
                TemporalAttentionBlock(temporal_dim, num_heads)
                for _ in range(num_attention_blocks)
            ]
        )

        self.temporal_pool = nn.Sequential(
            nn.Conv1d(temporal_dim, temporal_dim, 1), nn.ReLU(inplace=True)
        )

        self.final_projection = nn.Sequential(
            nn.Linear(temporal_dim * 3, temporal_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(temporal_dim, embedding_dim),
        )

        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def encode_frames(self, frames):
        """Encode each frame individually."""
        # frames: (B, T, C, H, W)
        B, T, C, H, W = frames.shape

        frames_flat = rearrange(frames, "b t c h w -> (b t) c h w")
        spatial_features = self.spatial_encoder(frames_flat)
        spatial_features = rearrange(spatial_features, "(b t) d -> b t d", b=B, t=T)

        return spatial_features

    def temporal_encoding(self, features):
        """Temporal encoding with attention."""
        x = self.temporal_projection(features)

        x = self.pos_encoding(x)

        for conv_block in self.temporal_conv_blocks:
            x_conv = rearrange(x, "b t c -> b c t")
            x_conv = conv_block(x_conv)
            x_conv = rearrange(x_conv, "b c t -> b t c")
            x = x + x_conv

        for attn_block in self.attention_blocks:
            x = attn_block(x)

        return x

    def adaptive_pooling(self, features):
        """Adaptive pooling for managing variable-length sequences."""
        # features: (B, T, C)

        avg_pool = reduce(features, "b t c -> b c", "mean")

        max_pool = reduce(features, "b t c -> b c", "max")

        x_conv = rearrange(features, "b t c -> b c t")
        attention_weights = F.softmax(self.temporal_pool(x_conv), dim=2)
        weighted_pool = (x_conv * attention_weights).sum(dim=2)

        pooled = torch.cat([avg_pool, max_pool, weighted_pool], dim=1)

        return pooled

    def forward(self, video, return_features=False):
        """
        Forward pass for a complete video.

        Args:
            video: Tensor of shape (B, T, C, H, W) or (B, C, T, H, W)
            return_features: If True, also returns temporal features

        Returns:
            embedding: Tensor of shape (B, embedding_dim) normalized with L2
        """
        if video.dim() == 5 and video.shape[1] == 3:  # (B, C, T, H, W)
            video = rearrange(video, "b c t h w -> b t c h w")

        spatial_features = self.encode_frames(video)

        temporal_features = self.temporal_encoding(spatial_features)

        pooled_features = self.adaptive_pooling(temporal_features)

        embedding = self.final_projection(pooled_features)

        embedding = F.normalize(embedding, p=2, dim=1)

        if return_features:
            return embedding, temporal_features
        return embedding

    def compute_loss(self, video1, video2, extract_ratio=0.5):
        """
        Compute the contrastive loss with augmentation by extracting segments.

        Args:
            video1, video2: Pairs of videos (B, T, C, H, W)
            extract_ratio: Minimum ratio of frames to extract (0.5 = at least half)
        """
        B, T, C, H, W = video1.shape

        emb_full_1 = self.forward(video1)
        emb_full_2 = self.forward(video2)

        extracts_1, extracts_2 = [], []

        for b in range(B):
            extract_len = torch.randint(int(T * extract_ratio), T + 1, (1,)).item()

            if extract_len < T:
                start_1 = torch.randint(0, T - extract_len + 1, (1,)).item()
                start_2 = torch.randint(0, T - extract_len + 1, (1,)).item()
            else:
                start_1 = start_2 = 0

            extract_1 = video1[b : b + 1, start_1 : start_1 + extract_len]
            extract_2 = video2[b : b + 1, start_2 : start_2 + extract_len]

            extracts_1.append(extract_1)
            extracts_2.append(extract_2)

        emb_extract_1 = torch.cat([self.forward(e) for e in extracts_1], dim=0)
        emb_extract_2 = torch.cat([self.forward(e) for e in extracts_2], dim=0)

        logits_full = torch.matmul(emb_full_1, emb_full_2.T) / self.temperature
        labels = torch.arange(B, device=logits_full.device)
        loss_full = F.cross_entropy(logits_full, labels) + F.cross_entropy(
            logits_full.T, labels
        )

        logits_extract_1 = torch.matmul(emb_extract_1, emb_full_1.T) / self.temperature
        logits_extract_2 = torch.matmul(emb_extract_2, emb_full_2.T) / self.temperature
        loss_extract = F.cross_entropy(logits_extract_1, labels) + F.cross_entropy(
            logits_extract_2, labels
        )

        logits_extract_cross = (
            torch.matmul(emb_extract_1, emb_extract_2.T) / self.temperature
        )
        loss_extract_cross = F.cross_entropy(
            logits_extract_cross, labels
        ) + F.cross_entropy(logits_extract_cross.T, labels)

        total_loss = loss_full + 0.5 * loss_extract + 0.3 * loss_extract_cross

        return {
            "loss": total_loss / 3.6,
            "loss_full": loss_full / 2,
            "loss_extract": loss_extract / 2,
            "loss_extract_cross": loss_extract_cross / 2,
            "temperature": self.temperature,
        }


class Conv3DBlock(nn.Module):
    """Basic 3D convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class VideoFingerprint3D(nn.Module):
    """Fast 3D CNN model for video fingerprinting with temporal striding."""

    def __init__(self, embedding_dim=256, frame_stride=16, dropout=0.2):
        super().__init__()

        self.frame_stride = frame_stride
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            # First layer: aggressive temporal stride
            Conv3DBlock(
                3,
                64,
                kernel_size=(frame_stride, 5, 5),
                stride=(frame_stride, 2, 2),
                padding=(0, 2, 2),
            ),
            # Output: (B, 64, T/16, 32, 32)
            Conv3DBlock(
                64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
            ),
            # Output: (B, 128, T/32, 16, 16)
            Conv3DBlock(
                128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
            ),
            # Output: (B, 256, T/64, 8, 8)
            Conv3DBlock(
                256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
            ),
            # Output: (B, 512, T/128, 4, 4)
            # Final spatial reduction
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # Keep temporal dimension
            # Output: (B, 512, T/128, 1, 1)
        )

        # Temporal aggregation with learnable pooling
        self.temporal_conv = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.temporal_attention = nn.Conv1d(512, 1, kernel_size=1)

        # Final projection to embedding
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with better defaults for 3D convolutions."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, video):
        """Forward pass for video fingerprinting."""
        if video.dim() == 5 and video.shape[2] == 3:  # (B, T, C, H, W)
            video = rearrange(video, "b t c h w -> b c t h w")

        B, C, T, H, W = video.shape

        # Pad temporal dimension to be divisible by frame_stride
        pad_frames = (self.frame_stride - T % self.frame_stride) % self.frame_stride
        if pad_frames > 0:
            video = F.pad(video, (0, 0, 0, 0, 0, pad_frames), mode="constant", value=0)

        # 3D CNN encoding
        features = self.encoder(video)  # (B, 512, T', 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 512, T')

        # Temporal aggregation with attention
        temporal_features = self.temporal_conv(features)  # (B, 512, T')

        # Compute attention weights
        attention_weights = self.temporal_attention(temporal_features)  # (B, 1, T')
        attention_weights = F.softmax(attention_weights, dim=2)

        # Weighted pooling
        weighted_features = (temporal_features * attention_weights).sum(
            dim=2
        )  # (B, 512)

        # Also compute average pooling for robustness
        avg_features = temporal_features.mean(dim=2)  # (B, 512)

        # Combine both pooling methods
        combined_features = weighted_features + avg_features

        # Final projection
        embedding = self.projector(combined_features)

        # L2 normalization
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def compute_loss(self, video1, video2, hard_negative_ratio=0.3):
        """Compute contrastive loss between video pairs."""
        B = video1.shape[0]

        # Get embeddings
        emb1 = self.forward(video1)
        emb2 = self.forward(video2)

        # Compute similarity matrix
        logits = torch.matmul(emb1, emb2.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=logits.device)

        # Standard contrastive loss
        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.T, labels)

        # Hard negative mining
        with torch.no_grad():
            # Find hardest negatives (highest similarity among negatives)
            mask = torch.eye(B, device=logits.device).bool()
            neg_logits = logits.masked_fill(mask, float("-inf"))

            # Get top-k hard negatives
            k = max(1, int(B * hard_negative_ratio))
            hard_indices = torch.topk(neg_logits, k, dim=1).indices

        # Compute hard negative loss
        hard_logits = torch.gather(logits, 1, hard_indices)
        hard_labels = torch.zeros(B, dtype=torch.long, device=logits.device)

        # Create new logits with positive and hard negatives
        pos_logits = torch.diagonal(logits).unsqueeze(1)
        hard_loss_logits = torch.cat([pos_logits, hard_logits], dim=1)
        hard_loss = F.cross_entropy(hard_loss_logits, hard_labels)

        # Total loss
        total_loss = (loss_12 + loss_21) / 2 + 0.5 * hard_loss

        return {
            "loss": total_loss,
            "loss_standard": (loss_12 + loss_21) / 2,
            "loss_hard": hard_loss,
            "temperature": self.temperature.item(),
        }


def create_model(model_type="attention", **kwargs):
    """
    Create a video fingerprint model.

    Args:
        model_type: 'attention' or '3d' or 'cnn3d'
        **kwargs: Model-specific parameters

    Returns:
        Model instance
    """
    if model_type == "attention":
        return VideoFingerprintAttention(
            spatial_dim=kwargs.get("spatial_dim", 128),
            temporal_dim=kwargs.get("temporal_dim", 256),
            embedding_dim=kwargs.get("embedding_dim", 256),
            num_attention_blocks=kwargs.get("num_attention_blocks", 4),
        )
    elif model_type in ["3d", "cnn3d"]:
        return VideoFingerprint3D(
            embedding_dim=kwargs.get("embedding_dim", 256),
            frame_stride=kwargs.get("frame_stride", 16),
            dropout=kwargs.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
