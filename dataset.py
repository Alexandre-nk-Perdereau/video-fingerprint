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
        chunk_size=300,
        min_extract_ratio=0.5,
        augment=True,
        cache_videos=True,
        mode="train",
    ):
        self.video_dir = Path(video_dir)
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.chunk_size = chunk_size
        self.min_extract_ratio = min_extract_ratio
        self.augment = augment
        self.mode = mode
        self.cache_videos = cache_videos
        self._cache = {}

        self.video_paths = []
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            self.video_paths.extend(list(self.video_dir.glob(f"**/{ext}")))

        self.video_chunks = []
        for video_path in self.video_paths:
            self.video_chunks.append(
                {"path": video_path, "chunk_idx": 0, "video_id": len(self.video_chunks)}
            )

        print(f"Found {len(self.video_paths)} videos")

    def __len__(self):
        return len(self.video_chunks)

    def _load_video_full(self, video_path):
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

    def _resize_frame(self, frame):
        h, w = frame.shape[:2]

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
        brightness = random.uniform(0.8, 1.2) if do_color else 1.0
        contrast = random.uniform(0.8, 1.2) if do_color else 1.0
        saturation = random.uniform(0.8, 1.2) if do_color else 1.0

        do_flip = random.random() > 0.5
        do_noise = random.random() > 0.7
        noise_level = random.uniform(0.01, 0.05) if do_noise else 0

        do_compression = random.random() > 0.5
        jpeg_quality = random.randint(60, 95) if do_compression else 100

        augmented = []

        for frame in frames:
            aug_frame = frame.copy()

            if do_color:
                aug_frame = aug_frame.astype(np.float32) / 255.0

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

            if do_compression and random.random() > 0.5:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                _, encimg = cv2.imencode(
                    ".jpg", cv2.cvtColor(aug_frame, cv2.COLOR_RGB2BGR), encode_param
                )
                aug_frame = cv2.cvtColor(cv2.imdecode(encimg, 1), cv2.COLOR_BGR2RGB)

            augmented.append(aug_frame)

        return augmented

    def _create_extract_pair(self, frames):
        n_frames = len(frames)

        if self.mode == "train":
            min_length = int(n_frames * self.min_extract_ratio)

            len1 = random.randint(min_length, n_frames)
            len2 = random.randint(min_length, n_frames)

            start1 = random.randint(0, n_frames - len1)
            start2 = random.randint(0, n_frames - len2)

            if random.random() > 0.3:
                overlap_frames = random.randint(min_length // 2, min(len1, len2))
                max_offset = min(len1, len2) - overlap_frames
                offset = random.randint(-max_offset, max_offset)
                start2 = max(0, min(start1 + offset, n_frames - len2))

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

    def __getitem__(self, idx):
        chunk_info = self.video_chunks[idx]
        video_path = chunk_info["path"]

        all_frames = self._load_video_full(video_path)

        frames1, frames2 = self._create_extract_pair(all_frames)

        frames1 = [self._resize_frame(f) for f in frames1]
        frames2 = [self._resize_frame(f) for f in frames2]

        frames1 = self._apply_augmentations(frames1)
        frames2 = self._apply_augmentations(frames2)

        clip1 = torch.from_numpy(np.stack(frames1)).float() / 255.0
        clip2 = torch.from_numpy(np.stack(frames2)).float() / 255.0

        clip1 = clip1.permute(0, 3, 1, 2)
        clip2 = clip2.permute(0, 3, 1, 2)

        return {
            "clip1": clip1,  # (T1, C, H, W)
            "clip2": clip2,  # (T2, C, H, W)
            "video_id": chunk_info["video_id"],
            "lengths": torch.tensor([len(frames1), len(frames2)]),
        }


def collate_fn_padding(batch):
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

    clips1_batch = torch.stack(clips1_padded)  # (B, T, C, H, W)
    clips2_batch = torch.stack(clips2_padded)  # (B, T, C, H, W)

    return {"clip1": clips1_batch, "clip2": clips2_batch, "video_id": video_ids}


def create_dataloader(
    video_dir, batch_size=8, num_workers=4, frame_size=64, max_frames=500, mode="train"
):
    dataset = VideoFingerprintDataset(
        video_dir=video_dir,
        frame_size=frame_size,
        max_frames=max_frames,
        augment=(mode == "train"),
        mode=mode,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn_padding,
        pin_memory=True,
        drop_last=(mode == "train"),
    )

    return dataloader
