import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
from pathlib import Path
import av
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


class VideoFingerprintDataset(Dataset):
    """
    Dataset which loads frames from videos for fingerprinting.
    Handles videos of variable lengths.
    """

    def __init__(
        self,
        video_dir,
        frame_size=64,
        max_frames=1000,
        clip_length=128,  # For 3D CNN mode
        frame_stride=32,  # For 3D CNN mode
        min_extract_ratio=0.5,
        augment=True,
        cache_videos=True,
        mode="train",
        model_type="attention",  # 'attention' or '3d'
    ):
        self.video_dir = Path(video_dir)
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.min_extract_ratio = min_extract_ratio
        self.augment = augment
        self.mode = mode
        self.model_type = model_type
        self.cache_videos = cache_videos
        self._cache = {}

        self.video_paths = []
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            self.video_paths.extend(list(self.video_dir.glob(f"**/{ext}")))

        if model_type == "attention":
            self.samples = []
            for video_path in self.video_paths:
                self.samples.append({"path": video_path, "video_id": len(self.samples)})
        else:
            self._create_3d_clips_metadata()

        print(f"Found {len(self.video_paths)} videos")
        print(f"Dataset mode: {model_type}, Total samples: {len(self)}")

    def _create_3d_clips_metadata(self):
        """Create metadata for 3D CNN clips."""
        self.samples = []

        for video_id, video_path in enumerate(self.video_paths):
            try:
                container = av.open(str(video_path))
                stream = container.streams.video[0]
                total_frames = stream.frames

                if total_frames == 0:
                    total_frames = int(stream.duration * stream.average_rate)

                container.close()

                if total_frames >= self.clip_length:
                    if self.mode == "train":
                        num_clips = min(5, (total_frames - self.clip_length) // 32 + 1)
                        for i in range(num_clips):
                            self.samples.append(
                                {
                                    "path": video_path,
                                    "video_id": video_id,
                                    "total_frames": total_frames,
                                    "clip_idx": i,
                                }
                            )
                    else:
                        self.samples.append(
                            {
                                "path": video_path,
                                "video_id": video_id,
                                "total_frames": total_frames,
                                "clip_idx": 0,
                            }
                        )
                else:
                    self.samples.append(
                        {
                            "path": video_path,
                            "video_id": video_id,
                            "total_frames": total_frames,
                            "clip_idx": 0,
                        }
                    )

            except Exception as e:
                print(f"Error processing {video_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def _load_video_full(self, video_path):
        """Load all frames from a video (for attention model)."""
        if self.cache_videos and str(video_path) in self._cache:
            return self._cache[str(video_path)]

        frames = []

        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            stream.codec_context.skip_frame = "NONKEY"
            container.seek(0)
            total_frames = stream.frames

            if total_frames == 0:
                total_frames = int(stream.duration * stream.average_rate)

            # Simulate variable frame rate by changing skip rate
            if self.augment and self.mode == "train":
                speed_factor = random.uniform(0.5, 2.0)
                skip_rate = max(
                    1, int((total_frames // self.max_frames) * speed_factor)
                )
            else:
                skip_rate = max(1, total_frames // self.max_frames)

            container.seek(0)
            stream.codec_context.skip_frame = "DEFAULT"

            frame_count = 0
            for i, frame in enumerate(container.decode(stream)):
                if i % skip_rate == 0:
                    img = frame.to_ndarray(format="rgb24")
                    frames.append(img)
                    frame_count += 1

                    if frame_count >= self.max_frames:
                        break

            container.close()

        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(30)]

        if self.cache_videos and len(self._cache) < 100:
            self._cache[str(video_path)] = frames

        return frames

    def _load_clip_frames(self, video_path, start_frame, num_frames):
        """Load a contiguous clip of frames (for 3D CNN model)."""
        frames = []

        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            stream.codec_context.skip_frame = "NONKEY"
            container.seek(start_frame, stream=stream)
            stream.codec_context.skip_frame = "DEFAULT"

            frame_count = 0
            for frame in container.decode(stream):
                if frame_count >= num_frames:
                    break

                img = frame.to_ndarray(format="rgb24")
                frames.append(img)
                frame_count += 1

            container.close()

        except Exception as e:
            print(f"Error loading clip from {video_path}: {e}")
            frames = [
                np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(num_frames)
            ]

        while len(frames) < num_frames:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

        return frames[:num_frames]

    def _resize_frame(self, frame, apply_resolution_change=False):
        """Resize frame with optional resolution degradation."""
        h, w = frame.shape[:2]

        if apply_resolution_change and self.augment and random.random() > 0.5:
            resolutions = [
                (480, 640),
                (720, 1280),
                (1080, 1920),
                (360, 640),
            ]
            target_h, target_w = random.choice(resolutions)

            if h > target_h or w > target_w:
                scale = min(target_h / h, target_w / w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w

        if self.augment and random.random() > 0.3:
            crop_ratio = random.uniform(0.8, 1.0)
            crop_h = int(h * crop_ratio)
            crop_w = int(w * crop_ratio)

            start_h = random.randint(0, h - crop_h)
            start_w = random.randint(0, w - crop_w)

            frame = frame[start_h : start_h + crop_h, start_w : start_w + crop_w]
            h, w = crop_h, crop_w

        if h < w:
            new_h = self.frame_size
            new_w = int(w * self.frame_size / h)
        else:
            new_w = self.frame_size
            new_h = int(h * self.frame_size / w)

        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = frame.shape[:2]
        start_h = (h - self.frame_size) // 2
        start_w = (w - self.frame_size) // 2

        frame_cropped = frame[
            start_h : start_h + self.frame_size, start_w : start_w + self.frame_size
        ]

        return frame_cropped

    def _apply_augmentations(self, frames):
        if not self.augment:
            return frames

        do_color = random.random() > 0.3
        do_flip = random.random() > 0.5
        do_noise = random.random() > 0.7
        do_compression = random.random() > 0.5
        do_blur = random.random() > 0.5
        do_letterbox = random.random() > 0.7
        do_overlay = random.random() > 0.8
        do_rotation = random.random() > 0.8

        brightness = random.uniform(0.5, 1.5) if do_color else 1.0
        contrast = random.uniform(0.5, 1.5) if do_color else 1.0
        saturation = random.uniform(0.5, 1.5) if do_color else 1.0
        hue_shift = random.uniform(-0.1, 0.1) if do_color else 0

        noise_level = random.uniform(0.02, 0.1) if do_noise else 0
        jpeg_quality = random.randint(30, 90) if do_compression else 100
        blur_kernel = random.choice([3, 5, 7]) if do_blur else 0

        augmented = []

        for frame in frames:
            aug_frame = frame.copy()

            if do_color:
                aug_frame = aug_frame.astype(np.float32) / 255.0

                hsv = cv2.cvtColor(
                    (aug_frame * 255).astype(np.uint8), cv2.COLOR_RGB2HSV
                ).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
                aug_frame = (
                    cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(
                        np.float32
                    )
                    / 255.0
                )

                aug_frame = aug_frame * brightness

                aug_frame = (aug_frame - 0.5) * contrast + 0.5

                gray = cv2.cvtColor(
                    (aug_frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                )
                gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
                aug_frame = saturation * aug_frame + (1 - saturation) * gray

                aug_frame = np.clip(aug_frame * 255, 0, 255).astype(np.uint8)

            if do_flip:
                aug_frame = cv2.flip(aug_frame, 1)

            if do_noise:
                noise = np.random.randn(*aug_frame.shape) * noise_level * 255
                aug_frame = np.clip(
                    aug_frame.astype(np.float32) + noise, 0, 255
                ).astype(np.uint8)

            if do_blur:
                aug_frame = cv2.GaussianBlur(aug_frame, (blur_kernel, blur_kernel), 0)

            if do_compression:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                _, encimg = cv2.imencode(
                    ".jpg", cv2.cvtColor(aug_frame, cv2.COLOR_RGB2BGR), encode_param
                )
                aug_frame = cv2.cvtColor(cv2.imdecode(encimg, 1), cv2.COLOR_BGR2RGB)

            if do_letterbox:
                bar_size = random.randint(5, 15)  # pixels
                if random.random() > 0.5:  # Top and bottom
                    aug_frame[:bar_size, :] = 0
                    aug_frame[-bar_size:, :] = 0
                else:  # Left and right
                    aug_frame[:, :bar_size] = 0
                    aug_frame[:, -bar_size:] = 0

            if do_overlay:
                overlay_h = random.randint(10, 20)
                overlay_w = random.randint(30, 60)
                overlay_y = random.randint(0, self.frame_size - overlay_h)
                overlay_x = random.randint(0, self.frame_size - overlay_w)

                # Semi-transparent white rectangle
                overlay_region = aug_frame[
                    overlay_y : overlay_y + overlay_h, overlay_x : overlay_x + overlay_w
                ]
                white_overlay = np.ones_like(overlay_region) * 255
                alpha = 0.3
                aug_frame[
                    overlay_y : overlay_y + overlay_h, overlay_x : overlay_x + overlay_w
                ] = (1 - alpha) * overlay_region + alpha * white_overlay

            if do_rotation:
                angle = random.uniform(-5, 5)  # degrees
                center = (self.frame_size // 2, self.frame_size // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aug_frame = cv2.warpAffine(
                    aug_frame, M, (self.frame_size, self.frame_size)
                )

            augmented.append(aug_frame)

        return augmented

    def _create_extract_pair(self, frames):
        """Create a pair of video extracts (for attention model)."""
        n_frames = len(frames)

        if self.mode == "train":
            min_length = int(n_frames * self.min_extract_ratio)

            len1 = random.randint(min_length, n_frames)
            len2 = random.randint(min_length, n_frames)

            start1 = random.randint(0, n_frames - len1)

            duplicate_type = random.random()

            if duplicate_type < 0.33:  # 33% - exact temporal overlap
                start2 = start1
                len2 = len1
            elif duplicate_type < 0.66:  # 33% - partial overlap
                overlap_frames = random.randint(min_length // 3, min(len1, len2) // 2)
                max_offset = min(len1, len2) - overlap_frames
                offset = random.randint(-max_offset, max_offset)
                start2 = max(0, min(start1 + offset, n_frames - len2))
            else:  # 33% - simulate trimmed/extended versions
                if random.random() > 0.5:
                    # Trimmed version
                    start2 = start1 + random.randint(0, max(1, len1 // 4))
                    len2 = len1 - random.randint(0, max(1, len1 // 4))
                else:
                    # Extended version
                    start2 = max(0, start1 - random.randint(0, max(1, len1 // 4)))
                    len2 = min(
                        n_frames - start2, len1 + random.randint(0, max(1, len1 // 4))
                    )

            extract1 = frames[start1 : start1 + len1]
            extract2 = frames[start2 : start2 + len2]

        else:  # validation/test
            extract1 = frames

            extract_len = random.randint(
                int(n_frames * self.min_extract_ratio), n_frames
            )
            start = random.randint(0, n_frames - extract_len)
            extract2 = frames[start : start + extract_len]

        return extract1, extract2

    def _get_clip_start_position(self, clip_info):
        """Determine the starting frame for a clip (for 3D CNN model)."""
        total_frames = clip_info["total_frames"]

        if total_frames <= self.clip_length:
            return 0

        if self.mode == "train":
            max_start = total_frames - self.clip_length
            return random.randint(0, max_start)
        else:
            clip_idx = clip_info["clip_idx"]
            if clip_idx == 0:
                return (total_frames - self.clip_length) // 2
            else:
                num_positions = 5
                position = clip_idx % num_positions
                return (
                    position * (total_frames - self.clip_length) // (num_positions - 1)
                )

    def __getitem__(self, idx):
        if self.model_type == "attention":
            return self._get_attention_item(idx)
        else:
            return self._get_3d_item(idx)

    def _get_attention_item(self, idx):
        """Get item for attention model."""
        sample_info = self.samples[idx]
        video_path = sample_info["path"]

        all_frames = self._load_video_full(video_path)
        frames1, frames2 = self._create_extract_pair(all_frames)

        frames1 = [self._resize_frame(f, apply_resolution_change=True) for f in frames1]
        frames2 = [self._resize_frame(f, apply_resolution_change=True) for f in frames2]

        frames1 = self._apply_augmentations(frames1)
        frames2 = self._apply_augmentations(frames2)

        clip1 = torch.from_numpy(np.stack(frames1)).float() / 255.0
        clip2 = torch.from_numpy(np.stack(frames2)).float() / 255.0

        clip1 = clip1.permute(0, 3, 1, 2)
        clip2 = clip2.permute(0, 3, 1, 2)

        return {
            "clip1": clip1,
            "clip2": clip2,
            "video_id": sample_info["video_id"],
            "lengths": torch.tensor([len(frames1), len(frames2)]),
        }

    def _get_3d_item(self, idx):
        """Get item for 3D CNN model."""
        clip_info = self.samples[idx]
        video_path = clip_info["path"]

        start1 = self._get_clip_start_position(clip_info)
        start2 = self._get_clip_start_position(clip_info)

        if self.mode == "train":
            duplicate_type = random.random()
            if duplicate_type < 0.4:  # Exact same clip
                start2 = start1
            else:  # Overlapping clips
                max_offset = self.clip_length // 3
                offset = random.randint(-max_offset, max_offset)
                start2 = max(
                    0,
                    min(start1 + offset, clip_info["total_frames"] - self.clip_length),
                )

        frames1 = self._load_clip_frames(video_path, start1, self.clip_length)
        frames2 = self._load_clip_frames(video_path, start2, self.clip_length)

        frames1 = [self._resize_frame(f, apply_resolution_change=True) for f in frames1]
        frames2 = [self._resize_frame(f, apply_resolution_change=True) for f in frames2]

        frames1 = self._apply_augmentations(frames1)
        frames2 = self._apply_augmentations(frames2)

        clip1 = torch.from_numpy(np.stack(frames1)).float() / 255.0
        clip2 = torch.from_numpy(np.stack(frames2)).float() / 255.0

        clip1 = clip1.permute(0, 3, 1, 2)
        clip2 = clip2.permute(0, 3, 1, 2)

        return {"clip1": clip1, "clip2": clip2, "video_id": clip_info["video_id"]}


def collate_fn_padding(batch):
    """Collate function for attention model with padding."""
    clips1 = [item["clip1"] for item in batch]
    clips2 = [item["clip2"] for item in batch]
    video_ids = torch.tensor([item["video_id"] for item in batch])

    max_len1 = max(clip.shape[0] for clip in clips1)
    max_len2 = max(clip.shape[0] for clip in clips2)

    clips1_padded = []
    clips2_padded = []

    for clip in clips1:
        T, C, H, W = clip.shape
        if T < max_len1:
            padding = torch.zeros(max_len1 - T, C, H, W)
            clip_padded = torch.cat([clip, padding], dim=0)
        else:
            clip_padded = clip
        clips1_padded.append(clip_padded)

    for clip in clips2:
        T, C, H, W = clip.shape
        if T < max_len2:
            padding = torch.zeros(max_len2 - T, C, H, W)
            clip_padded = torch.cat([clip, padding], dim=0)
        else:
            clip_padded = clip
        clips2_padded.append(clip_padded)

    clips1_batch = torch.stack(clips1_padded)
    clips2_batch = torch.stack(clips2_padded)

    return {"clip1": clips1_batch, "clip2": clips2_batch, "video_id": video_ids}


def create_dataloader(
    video_dir,
    batch_size=8,
    num_workers=4,
    frame_size=64,
    max_frames=500,
    clip_length=128,
    frame_stride=16,
    mode="train",
    model_type="attention",
):
    """
    Create a dataloader for video fingerprinting.

    Args:
        video_dir: Path to video directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        frame_size: Target frame size
        max_frames: Max frames for attention model
        clip_length: Clip length for 3D model
        frame_stride: Frame stride for 3D model
        mode: 'train' or 'val'
        model_type: 'attention' or '3d'
    """
    dataset = VideoFingerprintDataset(
        video_dir=video_dir,
        frame_size=frame_size,
        max_frames=max_frames,
        clip_length=clip_length,
        frame_stride=frame_stride,
        augment=(mode == "train"),
        mode=mode,
        model_type=model_type,
    )

    collate_fn = collate_fn_padding if model_type == "attention" else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(mode == "train"),
    )

    return dataloader
