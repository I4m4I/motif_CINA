from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from .config import get_config
from .tools import (
    RNNClassifier,
    build_diag_mask,
    evaluate,
    load_dataset,
    load_trained_model,
    motifRegular,
    motif_initialisation,
    noise_sweep,
    save_model,
    train_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training and noise evaluation pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model on a selected dataset")
    train_parser.add_argument("--dataset", choices=["smnist", "fmnist", "tidigits"], required=True)
    train_parser.add_argument("--run-id", type=str, required=True, help="Identifier for saving the model")
    train_parser.add_argument("--fre", nargs=13, type=float, default=None, help="Override motif frequencies (13 floats)")
    train_parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device preference")
    train_parser.add_argument("--prefix", type=str, default="TEA", help="Optional prefix for saved model files")

    noise_parser = subparsers.add_parser("noise", help="Evaluate noise robustness for a trained model")
    noise_parser.add_argument("--dataset", choices=["smnist", "fmnist", "tidigits"], required=True)
    noise_parser.add_argument("--run-id", type=str, required=True, help="Identifier used during training")
    noise_parser.add_argument("--max-var", type=float, default=1.0, help="Maximum Gaussian noise variance")
    noise_parser.add_argument("--steps", type=int, default=26, help="Number of points in the variance sweep")
    noise_parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device preference")
    noise_parser.add_argument("--prefix", type=str, default="TEA", help="Optional prefix used when saving the model")
    noise_parser.add_argument("--discrete", action="store_true", help="Quantise tanh activations during the discrete inference path")
    noise_parser.add_argument(
        "--discrete-levels",
        "--discrete-num",
        dest="discrete_levels",
        type=int,
        default=None,
        help="Number of quantisation levels in [-1, 1] used when --discrete is enabled",
    )

    return parser.parse_args()


def resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def run_train(args: argparse.Namespace) -> None:
    cfg = get_config(args.dataset)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    fre_values = args.fre if args.fre is not None else cfg.fre
    if len(fre_values) != 13:
        raise ValueError("fre must contain exactly 13 values")
    skip_motif = args.fre is not None and all(val < 0 for val in fre_values)

    data = load_dataset(cfg.name, cfg.batch_size, validation_split=cfg.validation_split, dataset_kwargs=cfg.dataset_kwargs)
    model = RNNClassifier(
        data.input_size,
        cfg.num_neurons,
        data.num_classes,
        num_layers=1,
        diag_value=cfg.diag_value,
        bin_mode=cfg.bin_mode,
        threshold=cfg.threshold,
        n_levels=cfg.n_levels,
        diag_init=cfg.diag_bool,
    ).to(device)

    if skip_motif:
        print("Skipping motif initialisation because --fre entries are all negative.")
    else:
        motif_reg = motifRegular(fre_values, device=device, num_neuron=cfg.num_neurons, amplitude=cfg.amplitude, bias=cfg.bias)
        motif_opt = torch.optim.SGD(model.parameters(), lr=cfg.motif_lr, momentum=cfg.momentum)
        motif_initialisation(model, motif_reg, cfg.motif_epochs, motif_opt, scale=cfg.motif_loss_scale)

    diag_mask = build_diag_mask(cfg.num_neurons, device) if cfg.diag_enforce else None
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
    train_task(model, data.train, data.valid, cfg.train_epochs, optimiser, criterion, device, diag_mask=diag_mask)

    test_cont = evaluate(model, data.test, device, binary=False)
    test_disc = evaluate(model, data.test, device, binary=True)
    print(f"Test accuracy -> continuous: {test_cont:.2f}% | discrete: {test_disc:.2f}%")

    prefix = args.prefix.strip()
    base_name = f"{prefix}_{args.run_id}" if prefix else str(args.run_id)
    ckpt_path = Path(cfg.model_dir) / f"{base_name}.pt"
    save_model(model, ckpt_path)
    print(f"Model saved to {ckpt_path}")


def run_noise(args: argparse.Namespace) -> None:
    cfg = get_config(args.dataset)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if args.discrete_levels is not None and args.discrete_levels < 2:
        raise ValueError("--discrete-levels must be at least 2")

    data = load_dataset(cfg.name, cfg.batch_size, validation_split=cfg.validation_split, dataset_kwargs=cfg.dataset_kwargs)
    prefix = args.prefix.strip()
    base_name = f"{prefix}_{args.run_id}" if prefix else str(args.run_id)
    ckpt_path = Path(cfg.model_dir) / f"{base_name}.pt"
    if not ckpt_path.is_file():
        legacy_path = Path(cfg.model_dir) / f"{args.run_id}.pt"
        if legacy_path.is_file():
            print(f"Model {ckpt_path.name} not found, falling back to legacy file {legacy_path.name}.")
            ckpt_path = legacy_path
        else:
            raise FileNotFoundError(f"Model checkpoint {ckpt_path} not found.")
    model = load_trained_model(
        ckpt_path,
        input_size=data.input_size,
        output_size=data.num_classes,
        diag_value=cfg.diag_value,
        bin_mode=cfg.bin_mode,
        threshold=cfg.threshold,
        n_levels=cfg.n_levels,
        device=device,
    )

    variances = np.linspace(0.0, args.max_var, num=args.steps, endpoint=True, dtype=np.float32)
    discrete_levels = args.discrete_levels if args.discrete else None
    if args.discrete:
        levels_msg = discrete_levels if discrete_levels is not None else model.n_levels
        print(f"Discrete noise evaluation enabled with {levels_msg} levels.")
    results = noise_sweep(model, data.test, device, variances, discrete_levels=discrete_levels)

    cfg.noise_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_discrete" if args.discrete else ""
    out_path = Path(cfg.noise_dir) / f"{base_name}{suffix}.npy"
    np.save(out_path, results)
    print(f"Noise sweep results saved to {out_path} (columns: variance, acc_cont, acc_disc)")


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "noise":
        run_noise(args)
    else:  # pragma: no cover
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()

# python -m classified.main train --dataset fmnist --run-id 0 --fre -1 -1 -1 -1 -1 -1 -1 -1 0.0318 0.0050 0.1492 0.3420 0.0917
# python -m classified.main train --dataset tidigits --run-id 0 --fre -1 -1 -1 -1 -1 -1 -1 -1 0.0318 0.0050 0.1492 0.3420 0.0917
