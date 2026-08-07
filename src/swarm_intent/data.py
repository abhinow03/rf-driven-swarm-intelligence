"""
Synthetic swarm-trajectory generation, dataset assembly, and splitting.

Consolidates the data-generation code that was duplicated across the three
notebooks. Two correctness fixes versus the originals:

1. A single seeded ``np.random.Generator`` is threaded through ALL sampling
   (velocity, direction, dispersed offsets, noise). The originals seeded only
   the outer loop, so datasets were not actually reproducible.
2. Train/val/test normalisation uses TRAIN statistics only, applied to all
   splits (the original computed per-split stats — harmless on i.i.d. synthetic
   data, but wrong in principle and a bug if data ever stops being i.i.d.).

NOTE: ``generate_transition_sequence`` is reconstructed from the documented
behaviour in the notebook (cosine-ramp blend between two formations). Diff it
against the original notebook on first run if exact parity matters.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .config import Config, BASE_FORMATIONS, TRANSITION_CLASS
from .formations import get_formation_offsets


def generate_swarm_sequence(
    formation_type: str,
    n_timesteps: int = 50,
    dt: float = 0.5,
    spread: float = 1.0,
    noise_std: float = 0.5,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, list, dict]:
    """Generate one (n_timesteps, 6, 3) trajectory for a single formation."""
    if rng is None:
        rng = np.random.default_rng()

    centroid = np.array([0.0, 0.0, 100.0])
    angle = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(3.0, 8.0)
    velocity = np.array([speed * np.cos(angle), speed * np.sin(angle),
                         rng.uniform(-0.5, 0.5)])
    # Acceleration ported from upstream commit 9158b081 -- colinear with the
    # initial heading, magnitude/sign randomized, so velocity is no longer
    # constant for the whole trajectory.
    accel_mag = rng.uniform(-1.0, 1.0)
    acceleration = np.array([accel_mag * np.cos(angle), accel_mag * np.sin(angle),
                             rng.uniform(-0.1, 0.1)])

    base_offsets = get_formation_offsets(formation_type, spread, rng=rng)
    sequence = np.zeros((n_timesteps, 6, 3))
    labels = []

    for t in range(n_timesteps):
        velocity = velocity + acceleration * dt
        centroid = centroid + velocity * dt
        if formation_type == "converging":
            shrink = 1.0 - (0.9 * t / (n_timesteps - 1))
            offsets = base_offsets * shrink
        else:
            offsets = base_offsets
        drone_positions = centroid + offsets
        drone_positions = drone_positions + rng.normal(0.0, noise_std, size=(6, 3))
        sequence[t] = drone_positions
        labels.append(formation_type)

    meta = {"formation_type": formation_type, "speed": speed,
            "direction_angle": angle, "spread": spread,
            "noise_std": noise_std, "n_timesteps": n_timesteps, "dt": dt}
    return sequence, labels, meta


def generate_transition_sequence(
    formation_a: str,
    formation_b: str,
    n_timesteps: int = 50,
    dt: float = 0.5,
    spread: float = 1.0,
    noise_std: float = 0.5,
    blend_start: int = 20,
    blend_end: int = 30,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a sequence that morphs from formation_a to formation_b.

    Offsets are interpolated per-drone with a cosine ramp over
    [blend_start, blend_end] so the transition is physically smooth.
    """
    if rng is None:
        rng = np.random.default_rng()

    centroid = np.array([0.0, 0.0, 100.0])
    angle = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(3.0, 8.0)
    velocity = np.array([speed * np.cos(angle), speed * np.sin(angle),
                         rng.uniform(-0.5, 0.5)])
    # Acceleration ported from upstream commit 9158b081 -- see
    # generate_swarm_sequence's comment for the rationale.
    accel_mag = rng.uniform(-1.0, 1.0)
    acceleration = np.array([accel_mag * np.cos(angle), accel_mag * np.sin(angle),
                             rng.uniform(-0.1, 0.1)])

    off_a = get_formation_offsets(formation_a, spread, rng=rng)
    off_b = get_formation_offsets(formation_b, spread, rng=rng)
    sequence = np.zeros((n_timesteps, 6, 3))

    for t in range(n_timesteps):
        velocity = velocity + acceleration * dt
        centroid = centroid + velocity * dt
        if t <= blend_start:
            alpha = 0.0
        elif t >= blend_end:
            alpha = 1.0
        else:
            frac = (t - blend_start) / (blend_end - blend_start)
            alpha = 0.5 * (1 - np.cos(np.pi * frac))  # cosine ramp 0->1
        offsets = (1 - alpha) * off_a + alpha * off_b
        drone_positions = centroid + offsets
        drone_positions = drone_positions + rng.normal(0.0, noise_std, size=(6, 3))
        sequence[t] = drone_positions
    return sequence


def generate_dataset(
    cfg: Config,
    n_per_formation: int = 1000,
    n_timesteps: int = 50,
    dt: float = 0.5,
    include_transitions: bool = False,
    n_transition: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build the full dataset across all formations (+ optional transitions).

    Returns
    -------
    X : (N, n_timesteps, 6, 3)
    y : (N,)  integer labels
    names : list[str]  index -> formation name
    """
    rng = np.random.default_rng(cfg.seed)
    names = list(BASE_FORMATIONS)
    label_map = {name: i for i, name in enumerate(names)}

    seqs, labels = [], []
    for formation in BASE_FORMATIONS:
        for _ in range(n_per_formation):
            spread = rng.uniform(0.7, 1.5)
            noise = rng.uniform(0.3, 0.8)
            seq, _, _ = generate_swarm_sequence(
                formation, n_timesteps, dt, spread, noise, rng=rng
            )
            seqs.append(seq)
            labels.append(label_map[formation])

    if include_transitions and n_transition > 0:
        names.append(TRANSITION_CLASS)
        trans_label = label_map.setdefault(TRANSITION_CLASS, len(label_map))
        pairs = [(a, b) for a in BASE_FORMATIONS for b in BASE_FORMATIONS if a != b]
        for _ in range(n_transition):
            a, b = pairs[rng.integers(len(pairs))]
            spread = rng.uniform(0.7, 1.5)
            noise = rng.uniform(0.3, 0.8)
            seq = generate_transition_sequence(a, b, n_timesteps, dt, spread, noise, rng=rng)
            seqs.append(seq)
            labels.append(trans_label)

    return np.array(seqs), np.array(labels), names


def split_and_normalize(X: np.ndarray, y: np.ndarray, cfg: Config) -> dict:
    """Stratified 70/15/15 split + train-only normalisation.

    Returns a dict with X_train/X_val/X_test, y_*, and train_mean/train_std.
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=cfg.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=cfg.seed
    )
    # Normalisation stats from TRAIN ONLY, applied everywhere.
    train_mean = X_train.mean()
    train_std = X_train.std() + 1e-8
    norm = lambda a: (a - train_mean) / train_std
    return {
        "X_train": norm(X_train), "X_val": norm(X_val), "X_test": norm(X_test),
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "train_mean": float(train_mean), "train_std": float(train_std),
    }


def save_splits(splits: dict, cfg: Config) -> None:
    os.makedirs(cfg.data_dir, exist_ok=True)
    for k in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        np.save(os.path.join(cfg.data_dir, f"{k}.npy"), splits[k])
    np.save(os.path.join(cfg.data_dir, "norm_stats.npy"),
            np.array([splits["train_mean"], splits["train_std"]]))
