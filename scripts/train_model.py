"""Train the STGT swarm model on the saved splits.

Usage:
    python scripts/train_model.py --classes 8 --epochs 80
"""
import argparse

from swarm_intent.config import Config
from swarm_intent.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=7, choices=[7, 8])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--data-dir", default="swarm_data")
    args = ap.parse_args()

    cfg = Config(n_classes=args.classes, epochs=args.epochs, data_dir=args.data_dir)
    train(cfg)


if __name__ == "__main__":
    main()
