import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from datetime import datetime
import json
import sys
import time


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, run_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_dir = Path(run_dir)
        self.model_type = config.get("model_type", "attention")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.scaler = (
            torch.amp.GradScaler("cuda") if config.get("use_amp", True) else None
        )

        if self.model_type == "attention":
            param_groups = [
                {
                    "params": self.model.spatial_encoder.parameters(),
                    "lr": config["lr_spatial"],
                },
                {
                    "params": self.model.attention_blocks.parameters(),
                    "lr": config["lr_attention"],
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "spatial_encoder" not in n and "attention_blocks" not in n
                    ]
                },
            ]
        else:  # 3D model
            param_groups = self.model.parameters()

        self.optimizer = optim.AdamW(
            param_groups,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999),
        )

        if self.model_type == "3d":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config["epochs"],
                eta_min=config["learning_rate"] * 0.01,
            )
        else:
            num_training_steps = len(train_loader) * config["epochs"]
            num_warmup_steps = num_training_steps // 10

            def lr_lambda(step):
                if step < num_warmup_steps:
                    return float(step) / float(max(1, num_warmup_steps))
                progress = float(step - num_warmup_steps) / float(
                    max(1, num_training_steps - num_warmup_steps)
                )
                return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.run_dir / "tensorboard")

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.epoch = 0
        self.global_step = 0

        self._save_training_info()

    def _save_training_info(self):
        """Save training configuration and information."""
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        info_path = self.run_dir / "training_info.txt"
        with open(info_path, "w") as f:
            f.write(
                f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Device: {self.device}\n")
            f.write(f"Model type: {self.model_type}\n")
            f.write(
                f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}\n"
            )
            f.write(
                f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n"
            )
            f.write("\nModel Architecture:\n")
            if self.model_type == "attention":
                f.write(
                    f"  - Spatial dimension: {self.config.get('spatial_dim', 128)}\n"
                )
                f.write(
                    f"  - Temporal dimension: {self.config.get('temporal_dim', 256)}\n"
                )
                f.write(
                    f"  - Number of attention blocks: {self.config.get('num_attention_blocks', 4)}\n"
                )
            else:  # 3D model
                f.write(f"  - Frame stride: {self.config.get('frame_stride', 16)}\n")
                f.write(f"  - Clip length: {self.config.get('clip_length', 128)}\n")
            f.write(f"  - Embedding dimension: {self.config['embedding_dim']}\n")
            f.write("\nData Configuration:\n")
            f.write(f"  - Frame size: {self.config['frame_size']}\n")
            if self.model_type == "attention":
                f.write(f"  - Max frames: {self.config['max_frames']}\n")
            else:
                f.write(f"  - Clip length: {self.config['clip_length']}\n")
            f.write(f"  - Batch size: {self.config['batch_size']}\n")
            f.write(f"  - Number of training batches: {len(self.train_loader)}\n")
            f.write(f"  - Number of validation batches: {len(self.val_loader)}\n")
            f.write("\nCommand line arguments:\n")
            f.write(f"  {' '.join(sys.argv)}\n")

    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()

        metrics = {
            "loss": 0,
            "acc": 0,
            "time_per_batch": 0,
            "loss_triplet": 0,
            "num_triplets": 0,
        }

        if self.model_type == "attention":
            metrics.update(
                {
                    "loss_full": 0,
                    "loss_extract": 0,
                    "loss_extract_cross": 0,
                }
            )
        else:  # 3D model
            metrics.update(
                {
                    "loss_standard": 0,
                    "loss_hard": 0,
                }
            )

        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            start_time = time.time()

            clip1 = batch["clip1"].to(self.device)
            clip2 = batch["clip2"].to(self.device)
            video_ids = batch["video_id"].to(self.device)

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    if self.model_type == "attention":
                        output = self.model.compute_loss(
                            clip1,
                            clip2,
                            video_ids=video_ids,
                            extract_ratio=self.config["min_extract_ratio"],
                            use_triplet=True,
                            triplet_weight=self.config.get("triplet_weight", 0.3),
                        )
                    else:  # 3D model
                        output = self.model.compute_loss(
                            clip1,
                            clip2,
                            video_ids=video_ids,
                            use_triplet=True,
                            triplet_weight=self.config.get("triplet_weight", 0.3),
                        )
            else:
                if self.model_type == "attention":
                    output = self.model.compute_loss(
                        clip1,
                        clip2,
                        video_ids=video_ids,
                        extract_ratio=self.config["min_extract_ratio"],
                        use_triplet=True,
                        triplet_weight=self.config.get("triplet_weight", 0.3),
                    )
                else:  # 3D model
                    output = self.model.compute_loss(
                        clip1,
                        clip2,
                        video_ids=video_ids,
                        use_triplet=True,
                        triplet_weight=self.config.get("triplet_weight", 0.3),
                    )

            loss = output["loss"]

            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if self.model_type == "attention":
                self.scheduler.step()

            with torch.no_grad():
                emb1 = self.model(clip1)
                emb2 = self.model(clip2)
                logits = torch.matmul(emb1, emb2.T) / self.model.temperature
                preds = logits.argmax(dim=1)
                targets = torch.arange(logits.shape[0], device=self.device)
                acc = (preds == targets).float().mean()

                batch_time = time.time() - start_time
                metrics["time_per_batch"] += batch_time

                for key in output:
                    if key.startswith("loss"):
                        if key in metrics:
                            metrics[key] += output[key].item()
                metrics["acc"] += acc.item()
                metrics["loss_triplet"] += output.get("loss_triplet", 0).item()
                metrics["num_triplets"] += output.get("num_triplets", 0)
                num_batches += 1

                if self.model_type == "attention":
                    current_lr = self.scheduler.get_last_lr()[0]
                else:
                    current_lr = self.optimizer.param_groups[0]["lr"]

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{acc.item():.3f}",
                        "triplet": f"{output.get('loss_triplet', 0):.3f}",
                        "lr": f"{current_lr:.2e}",
                        "time": f"{batch_time:.2f}s",
                    }
                )

                if self.global_step % 10 == 0:
                    self.writer.add_scalar(
                        "Train/loss_step", loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "Train/acc_step", acc.item(), self.global_step
                    )
                    self.writer.add_scalar("Train/lr", current_lr, self.global_step)

                self.global_step += 1

        for key in metrics:
            metrics[key] /= num_batches

        return metrics

    def validate(self):
        self.model.eval()

        metrics = {
            "loss": 0,
            "acc": 0,
        }

        if self.model_type == "attention":
            metrics.update(
                {
                    "loss_full": 0,
                    "loss_extract": 0,
                    "loss_extract_cross": 0,
                }
            )
        else:  # 3D model
            metrics.update(
                {
                    "loss_standard": 0,
                    "loss_hard": 0,
                }
            )

        num_batches = 0

        all_embeddings = []
        all_video_ids = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                clip1 = batch["clip1"].to(self.device)
                clip2 = batch["clip2"].to(self.device)
                video_ids = batch["video_id"]

                if self.scaler:
                    with torch.amp.autocast("cuda"):
                        output = self.model.compute_loss(clip1, clip2)
                else:
                    output = self.model.compute_loss(clip1, clip2)

                emb1 = self.model(clip1)
                emb2 = self.model(clip2)

                logits = torch.matmul(emb1, emb2.T) / self.model.temperature
                preds = logits.argmax(dim=1)
                targets = torch.arange(logits.shape[0], device=self.device)
                acc = (preds == targets).float().mean()

                for key in output:
                    if key.startswith("loss"):
                        if key in metrics:
                            metrics[key] += output[key].item()
                metrics["acc"] += acc.item()
                num_batches += 1

                all_embeddings.extend([emb1.cpu(), emb2.cpu()])
                all_video_ids.extend(video_ids.tolist() * 2)

        for key in metrics:
            metrics[key] /= num_batches

        all_embeddings = torch.cat(all_embeddings, dim=0)
        retrieval_metrics = self._compute_retrieval_metrics(
            all_embeddings, all_video_ids
        )
        metrics.update(retrieval_metrics)

        if self.model_type == "attention":
            extract_metrics = self._test_extract_robustness()
            metrics.update(extract_metrics)

        return metrics

    def _compute_retrieval_metrics(self, embeddings, video_ids, k_values=[1, 5, 10]):
        embeddings = embeddings.numpy()
        video_ids = np.array(video_ids)
        n_videos = len(set(video_ids))

        similarities = np.dot(embeddings, embeddings.T)

        metrics = {}

        for k in k_values:
            if k > n_videos - 1:
                continue

            recalls = []

            for i in range(len(embeddings)):
                sim_scores = similarities[i].copy()
                sim_scores[i] = -np.inf

                top_k_indices = np.argpartition(sim_scores, -k)[-k:]
                top_k_ids = video_ids[top_k_indices]

                correct = np.any(top_k_ids == video_ids[i])
                recalls.append(correct)

            metrics[f"R@{k}"] = np.mean(recalls)

        aps = []
        for i in range(len(embeddings)):
            sim_scores = similarities[i].copy()
            sim_scores[i] = -np.inf
            sorted_indices = np.argsort(-sim_scores)
            sorted_ids = video_ids[sorted_indices]

            positives = sorted_ids == video_ids[i]
            if positives.sum() > 0:
                precisions = np.cumsum(positives) / (np.arange(len(positives)) + 1)
                ap = (precisions * positives).sum() / positives.sum()
                aps.append(ap)

        metrics["mAP"] = np.mean(aps) if aps else 0.0

        return metrics

    def _test_extract_robustness(self, num_tests=50):
        self.model.eval()

        similarities = []
        extract_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_tests // self.config["batch_size"]:
                    break

                clip = batch["clip1"].to(self.device)
                B, T, C, H, W = clip.shape

                emb_full = self.model(clip)

                for ratio in extract_ratios:
                    extract_len = int(T * ratio)
                    if extract_len >= T:
                        continue

                    start = (T - extract_len) // 2
                    extract = clip[:, start : start + extract_len]

                    emb_extract = self.model(extract)

                    sim = F.cosine_similarity(emb_full, emb_extract).mean()
                    similarities.append({"ratio": ratio, "similarity": sim.item()})

        metrics = {}
        for ratio in extract_ratios:
            sims = [s["similarity"] for s in similarities if s["ratio"] == ratio]
            if sims:
                metrics[f"extract_sim_{int(ratio * 100)}"] = np.mean(sims)

        return metrics

    def save_checkpoint(self, is_best=False, metrics=None):
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
            "metrics": metrics,
        }

        torch.save(checkpoint, self.checkpoint_dir / "last.pth")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            if metrics:
                metrics_path = self.checkpoint_dir / "best_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)

        if self.epoch % 5 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch}.pth")
            if metrics:
                metrics_path = self.checkpoint_dir / f"epoch_{self.epoch}_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)

    def _update_training_log(self, train_metrics, val_metrics, is_best):
        """Update the training log file with latest results."""
        log_path = self.run_dir / "training_log.txt"

        with open(log_path, "a") as f:
            if self.epoch == 0:
                f.write("\n" + "=" * 80 + "\n")
                f.write(
                    "Epoch | Train Loss | Train Acc | Val Loss | Val Acc | R@1  | R@5  | mAP  | Best\n"
                )
                f.write("-" * 80 + "\n")

            f.write(f"{self.epoch:5d} | ")
            f.write(f"{train_metrics['loss']:10.4f} | ")
            f.write(f"{train_metrics['acc']:9.3f} | ")
            f.write(f"{val_metrics['loss']:8.4f} | ")
            f.write(f"{val_metrics['acc']:7.3f} | ")
            f.write(f"{val_metrics.get('R@1', 0):4.2f} | ")
            f.write(f"{val_metrics.get('R@5', 0):4.2f} | ")
            f.write(f"{val_metrics.get('mAP', 0):4.2f} | ")
            f.write(f"{'V' if is_best else 'X'}\n")

    def train(self):
        print(f"Training on {self.device}")
        print(f"Model type: {self.model_type}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(
            f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )
        print(f"\nRun directory: {self.run_dir}")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        patience = self.config.get("patience", 10)
        patience_counter = 0

        for epoch in range(self.epoch, self.config["epochs"]):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.model_type == "3d":
                self.scheduler.step()

            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.3f}, Time/batch: {train_metrics['time_per_batch']:.2f}s"
            )
            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.3f}"
            )
            print(
                f"Val   - R@1: {val_metrics.get('R@1', 0):.3f}, R@5: {val_metrics.get('R@5', 0):.3f}, mAP: {val_metrics.get('mAP', 0):.3f}"
            )

            if self.model_type == "attention":
                print(
                    f"Extract Robustness - 50%: {val_metrics.get('extract_sim_50', 0):.3f}, 70%: {val_metrics.get('extract_sim_70', 0):.3f}"
                )

            for key, value in train_metrics.items():
                self.writer.add_scalar(f"Train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"Val/{key}", value, epoch)

            is_best = val_metrics["acc"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["acc"]
                self.best_val_loss = val_metrics["loss"]
                print(f"🏆 New best validation accuracy: {val_metrics['acc']:.3f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{patience}")

            all_metrics = {
                "train": train_metrics,
                "val": val_metrics,
                "epoch": epoch,
            }
            self.save_checkpoint(is_best, metrics=all_metrics)
            self._update_training_log(train_metrics, val_metrics, is_best)

            if epoch > 20 and val_metrics["acc"] < 0.5:
                print(
                    "WARNING: Low accuracy after 20 epochs, consider checking data/model"
                )

            if patience_counter >= patience:
                print(
                    f"\nEarly stopping triggered after {patience} epochs without improvement."
                )
                break

        self.writer.close()

        summary_path = self.run_dir / "training_summary.txt"
        with open(summary_path, "w") as f:
            f.write(
                f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Model type: {self.model_type}\n")
            f.write(f"Total epochs: {self.epoch + 1}\n")
            f.write(f"Best validation accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"Best validation loss: {self.best_val_loss:.4f}\n")
            f.write(f"Final checkpoint: {self.checkpoint_dir / 'last.pth'}\n")
            f.write(f"Best checkpoint: {self.checkpoint_dir / 'best.pth'}\n")

        print("\n✅ Training completed!")
        print(f"Results saved to: {self.run_dir}")


def setup_run_directory(base_dir="./runs", prefix=""):
    """Create a unique run directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{prefix}run_{timestamp}"
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    latest_link = Path(base_dir) / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train Video Fingerprint Model (Attention or 3D CNN)"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to video dataset"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data workers"
    )
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument(
        "--no_amp", action="store_true", help="Disable mixed precision training"
    )
    parser.add_argument(
        "--run_name", type=str, help="Custom run name (default: timestamp)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="attention",
        choices=["attention", "3d"],
        help="Model type to train (attention or 3d)",
    )
    parser.add_argument(
        "--clip_length", type=int, default=128, help="Clip length for 3D model"
    )
    parser.add_argument(
        "--frame_stride", type=int, default=16, help="Frame stride for 3D model"
    )
    parser.add_argument(
        '--triplet_weight', type=float, default=0.3,
        help='Weight for triplet loss (default: 0.3)'
    )
    parser.add_argument(
        '--triplet_margin', type=float, default=0.3,
        help='Margin for triplet loss (default: 0.3)'
    )

    args = parser.parse_args()

    if args.run_name:
        run_dir = Path("./runs") / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        model_prefix = "3d_" if args.model == "3d" else ""
        run_dir = setup_run_directory(prefix=model_prefix)

    batch_size = args.batch_size if args.model == "attention" else args.batch_size * 2

    lr = args.lr if args.model == "attention" else args.lr * 3

    config = {
        "batch_size": batch_size,
        "epochs": args.epochs,
        "learning_rate": lr,
        "lr_spatial": lr * 0.1,  # Only used for attention model
        "lr_attention": lr * 0.5,  # Only used for attention model
        "weight_decay": 1e-4,
        "frame_size": 64,
        "max_frames": 500,  # For attention model
        "clip_length": args.clip_length,  # For 3D model
        "frame_stride": args.frame_stride,  # For 3D model
        "embedding_dim": 256,
        "spatial_dim": 128,
        "temporal_dim": 256,
        "num_attention_blocks": 4,
        "min_extract_ratio": 0.5,
        "use_amp": not args.no_amp,
        "patience": args.patience,
        "data_dir": str(args.data_dir),
        "num_workers": args.num_workers,
        "model_type": args.model,
        "command_line": " ".join(sys.argv),
        'triplet_weight': args.triplet_weight,
        'triplet_margin': args.triplet_margin,
    }

    from model import create_model
    from dataset import create_dataloader

    model = create_model(
        model_type=args.model,
        spatial_dim=config["spatial_dim"],
        temporal_dim=config["temporal_dim"],
        embedding_dim=config["embedding_dim"],
        num_attention_blocks=config["num_attention_blocks"],
        frame_stride=config["frame_stride"],
    )

    train_loader = create_dataloader(
        args.data_dir,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        frame_size=config["frame_size"],
        max_frames=config["max_frames"],
        clip_length=config["clip_length"],
        frame_stride=config["frame_stride"],
        mode="train",
        model_type=args.model,
    )

    val_loader = create_dataloader(
        args.data_dir,
        batch_size=config["batch_size"] * 2
        if args.model == "attention"
        else config["batch_size"],
        num_workers=args.num_workers,
        frame_size=config["frame_size"],
        max_frames=config["max_frames"],
        clip_length=config["clip_length"],
        frame_stride=config["frame_stride"],
        mode="val",
        model_type=args.model,
    )

    trainer = Trainer(model, train_loader, val_loader, config, run_dir)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.epoch = checkpoint["epoch"]
        trainer.global_step = checkpoint["global_step"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        trainer.best_val_acc = checkpoint["best_val_acc"]
        print(f"Resumed from epoch {trainer.epoch}")

        with open(trainer.run_dir / "training_info.txt", "a") as f:
            f.write(f"\n\nResumed from checkpoint: {args.checkpoint}\n")
            f.write(f"Resumed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    trainer.train()


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Quick test mode...")
        sys.argv.extend(
            ["--data_dir", "./test_videos", "--batch_size", "2", "--epochs", "2"]
        )

    main()
