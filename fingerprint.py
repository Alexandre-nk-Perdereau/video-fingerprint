import torch
import numpy as np
from pathlib import Path
import av
import cv2
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import hashlib
from typing import List, Dict, Tuple, Optional


class VideoFingerprintScanner:
    """Videos scanner for extracting fingerprints and detecting duplicates."""
    # TODO: parallelize, can't batch because of variable frame lengths

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Args:
            model_path: Path to the trained model .pth file
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Loading model from {model_path}...")
        self.model, self.config = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.frame_size = self.config.get("frame_size", 64)
        self.max_frames = self.config.get("max_frames", 500)
        self.embedding_dim = self.config.get("embedding_dim", 256)

        print(f"Model loaded - Device: {self.device}")
        print(
            f"Parameters: frame_size={self.frame_size}, max_frames={self.max_frames}, embedding_dim={self.embedding_dim}"
        )

    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, dict]:
        """Loads the model and its configuration from the checkpoint."""
        from model import create_attention_model

        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint.get("config", {})

        model = create_attention_model(
            spatial_dim=config.get("spatial_dim", 128),
            temporal_dim=config.get("temporal_dim", 256),
            embedding_dim=config.get("embedding_dim", 256),
            num_attention_blocks=config.get("num_attention_blocks", 4),
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return model, config

    def _load_video_frames(
        self, video_path: Path, skip_rate: int = 1
    ) -> List[np.ndarray]:
        """Loads all frames from a video with optional subsampling."""
        frames = []

        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            total_frames = stream.frames
            if total_frames == 0:
                total_frames = (
                    int(stream.duration * stream.average_rate) if stream.duration else 0
                )

            if total_frames > self.max_frames:
                skip_rate = max(skip_rate, total_frames // self.max_frames)

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
            return []

        return frames

    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        processed_frames = []

        for frame in frames:
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

            processed_frames.append(frame_cropped)

        clip = torch.from_numpy(np.stack(processed_frames)).float() / 255.0
        # (T, H, W, C) to (T, C, H, W)
        clip = clip.permute(0, 3, 1, 2)

        return clip

    def extract_fingerprint(
        self, video_path: Path, num_segments: int = 3
    ) -> Optional[np.ndarray]:
        """Extract the fingerprint (embedding) from a video.

        Args:
            video_path: Path to the video file
            num_segments: Number of segments to extract for averaging
        Returns:
            embedding: Normalized numpy vector or None if error
        """
        frames = self._load_video_frames(video_path)

        if len(frames) < 10:
            print(f"Video too short: {video_path} ({len(frames)} frames)")
            return None

        embeddings = []

        with torch.no_grad():
            if len(frames) <= self.max_frames:
                clip = self._preprocess_frames(frames)
                clip = clip.unsqueeze(0).to(self.device)
                embedding = self.model(clip)
                embeddings.append(embedding.cpu().numpy())
            else:
                segment_length = min(self.max_frames, len(frames) // num_segments)

                for i in range(num_segments):
                    start = (
                        i * (len(frames) - segment_length) // (num_segments - 1)
                        if num_segments > 1
                        else 0
                    )
                    end = start + segment_length

                    segment_frames = frames[start:end]
                    clip = self._preprocess_frames(segment_frames)
                    clip = clip.unsqueeze(0).to(self.device)

                    embedding = self.model(clip)
                    embeddings.append(embedding.cpu().numpy())

        final_embedding = np.mean(np.vstack(embeddings), axis=0)

        return final_embedding.squeeze()

    def scan_directory(
        self, directory: Path, extensions: List[str] = None
    ) -> Dict[str, dict]:
        """
        Recursively scans a directory to extract fingerprints.

        Args:
            directory: Directory to scan
            extensions: File extensions to consider

        Returns:
            fingerprints: Dict {video_path: {embedding, metadata}}
        """
        if extensions is None:
            extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]

        video_paths = []
        for ext in extensions:
            video_paths.extend(directory.glob(f"**/*{ext}"))
            video_paths.extend(directory.glob(f"**/*{ext.upper()}"))

        video_paths = list(set(video_paths))
        print(f"\n{len(video_paths)} videos found in {directory}")

        fingerprints = {}
        failed = 0

        for video_path in tqdm(video_paths, desc="Extracting fingerprints"):
            embedding = self.extract_fingerprint(video_path)

            if embedding is not None:
                file_hash = self._compute_file_hash(video_path, max_bytes=1024 * 1024)

                fingerprints[str(video_path)] = {
                    "embedding": embedding,
                    "path": str(video_path),
                    "name": video_path.name,
                    "size": video_path.stat().st_size,
                    "file_hash": file_hash,
                    "embedding_norm": np.linalg.norm(embedding),
                }
            else:
                failed += 1

        print(f"{len(fingerprints)} fingerprints extracted ({failed} failures)")
        return fingerprints

    def _compute_file_hash(self, file_path: Path, max_bytes: int = None) -> str:
        """Compute the MD5 hash of a file (or its first bytes)."""
        md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            if max_bytes:
                data = f.read(max_bytes)
                md5.update(data)
            else:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5.update(chunk)

        return md5.hexdigest()

    def find_duplicates(
        self,
        fingerprints: Dict[str, dict],
        similarity_threshold: float = 0.95,
        use_faiss: bool = True,
    ) -> List[List[dict]]:
        if len(fingerprints) < 2:
            return []

        print(f"\nSearching for duplicates (threshold: {similarity_threshold})...")

        paths = list(fingerprints.keys())
        embeddings = np.array([fingerprints[p]["embedding"] for p in paths]).astype(
            "float32"
        )

        if use_faiss and len(embeddings) > 100:
            duplicate_groups = self._find_duplicates_faiss(
                embeddings, paths, fingerprints, similarity_threshold
            )
        else:
            duplicate_groups = self._find_duplicates_direct(
                embeddings, paths, fingerprints, similarity_threshold
            )

        for group in duplicate_groups:
            hashes = [item["file_hash"] for item in group]
            for i, item in enumerate(group):
                item["exact_duplicate"] = hashes.count(item["file_hash"]) > 1

        return duplicate_groups

    def _find_duplicates_direct(
        self,
        embeddings: np.ndarray,
        paths: List[str],
        fingerprints: Dict[str, dict],
        threshold: float,
    ) -> List[List[dict]]:
        n = len(embeddings)
        processed = set()
        duplicate_groups = []

        similarities = np.dot(embeddings, embeddings.T)

        for i in range(n):
            if i in processed:
                continue

            similar_indices = np.where(similarities[i] >= threshold)[0]

            if len(similar_indices) > 1:
                group = []
                for idx in similar_indices:
                    if idx not in processed:
                        processed.add(idx)
                        item = fingerprints[paths[idx]].copy()
                        item["similarity"] = float(similarities[i, idx])
                        group.append(item)

                if len(group) > 1:
                    duplicate_groups.append(group)

        return duplicate_groups

    def _find_duplicates_faiss(
        self,
        embeddings: np.ndarray,
        paths: List[str],
        fingerprints: Dict[str, dict],
        threshold: float,
    ) -> List[List[dict]]:
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        k = min(20, len(embeddings))
        similarities, indices = index.search(embeddings, k)

        processed = set()
        duplicate_groups = []

        for i in range(len(embeddings)):
            if i in processed:
                continue

            group = []
            for j, (sim, idx) in enumerate(zip(similarities[i], indices[i])):
                if sim >= threshold and idx not in processed:
                    processed.add(idx)
                    item = fingerprints[paths[idx]].copy()
                    item["similarity"] = float(sim)
                    group.append(item)

            if len(group) > 1:
                duplicate_groups.append(group)

        return duplicate_groups

    def save_results(
        self,
        fingerprints: Dict[str, dict],
        duplicate_groups: List[List[dict]],
        output_path: Path,
    ):
        fingerprints_json = {}
        for path, data in fingerprints.items():
            data_copy = data.copy()
            data_copy["embedding"] = data_copy["embedding"].tolist()
            fingerprints_json[path] = data_copy

        results = {
            "metadata": {
                "scan_date": datetime.now().isoformat(),
                "total_videos": len(fingerprints),
                "duplicate_groups": len(duplicate_groups),
                "model_config": self.config,
            },
            "fingerprints": fingerprints_json,
            "duplicate_groups": duplicate_groups,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")

    def print_duplicate_report(self, duplicate_groups: List[List[dict]]):
        if not duplicate_groups:
            print("\nNo duplicates found!")
            return

        print(f"\nDUPLICATE REPORT")
        print(f"{'=' * 80}")
        print(f"Number of duplicate groups: {len(duplicate_groups)}")

        total_videos = sum(len(group) for group in duplicate_groups)
        print(f"Total number of duplicate videos: {total_videos}")

        total_size = 0
        savings = 0

        for group in duplicate_groups:
            group_size = sum(item["size"] for item in group)
            total_size += group_size
            savings += group_size - min(item["size"] for item in group)

        print(f"Total duplicate space: {self._format_size(total_size)}")
        print(f"Potential space savings: {self._format_size(savings)}")
        print(f"{'=' * 80}\n")

        for i, group in enumerate(duplicate_groups, 1):
            print(f"🎬 Group {i} ({len(group)} videos)")

            group_sorted = sorted(group, key=lambda x: x["size"], reverse=True)

            for j, item in enumerate(group_sorted):
                exact = "✓" if item.get("exact_duplicate") else " "
                print(f"  [{exact}] {Path(item['path']).name}")
                print(f"      {Path(item['path']).parent}")
                print(f"      Size: {self._format_size(item['size'])}")
                print(f"      Similarity: {item['similarity']:.3f}")
                if j == 0:
                    print(f"      #️Hash: {item['file_hash'][:16]}...")
                print()

            print(
                f"  Potential savings: {self._format_size(sum(item['size'] for item in group[1:]))}"
            )
            print(f"{'-' * 80}\n")

    def _format_size(self, size_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Video fingerprint scanner and duplicate detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s --model model.pth --scan /path/to/videos
  %(prog)s --model model.pth --scan /videos --threshold 0.9
  %(prog)s --model model.pth --scan /videos --output results.json
        """,
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained .pth model"
    )
    parser.add_argument(
        "--scan", type=str, required=True, help="Folder containing videos to scan"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for duplicates (0-1, default: 0.95)",
    )
    parser.add_argument("--output", type=str, help="JSON file to save the results")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".mp4", ".avi", ".mov", ".mkv"],
        help="Video file extensions to scan",
    )

    args = parser.parse_args()

    print(f"Starting video fingerprint scanner")
    print(f"{'=' * 80}")

    scanner = VideoFingerprintScanner(args.model, device=args.device)

    video_dir = Path(args.scan)
    if not video_dir.exists():
        print(f"Error: Folder {video_dir} does not exist")
        return 1

    fingerprints = scanner.scan_directory(video_dir, extensions=args.extensions)

    if not fingerprints:
        print("No videos could be analyzed")
        return 1

    duplicate_groups = scanner.find_duplicates(
        fingerprints, similarity_threshold=args.threshold
    )

    scanner.print_duplicate_report(duplicate_groups)

    if args.output:
        output_path = Path(args.output)
        scanner.save_results(fingerprints, duplicate_groups, output_path)

    print(f"\nScan complete!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
