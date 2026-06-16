from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    fre: List[float]
    diag_value: float
    diag_bool: bool
    num_neurons: int
    batch_size: int
    amplitude: float
    bias: float
    motif_epochs: int
    train_epochs: int
    motif_loss_scale: float
    motif_lr: float
    train_lr: float
    momentum: float = 0.5
    validation_split: float = 0.1
    bin_mode: str = "sign"
    n_levels: int = 16
    threshold: float = 0.0
    diag_enforce: bool = True
    model_dir: Path = field(default_factory=lambda: OUTPUT_DIR / "models")
    noise_dir: Path = field(default_factory=lambda: OUTPUT_DIR / "noise")
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "smnist": DatasetConfig(
        name="smnist",
        num_classes=10,
        fre=[0.3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        diag_value=-0.4,
        diag_bool=True,
        num_neurons=512,
        batch_size=256,
        amplitude=1000.0,
        bias=0.05,
        motif_epochs=120,
        train_epochs=20,
        motif_loss_scale=1e5,
        motif_lr=0.01,
        train_lr=0.001,
        validation_split=0.1,
        model_dir=OUTPUT_DIR / "smnist" / "models",
        noise_dir=OUTPUT_DIR / "smnist" / "noise_eval",
    ),
    "fmnist": DatasetConfig(
        name="fmnist",
        num_classes=10,
        fre=[0.032, 0.379, 0.035, -1, -1, -1, 0.041, -1, -1, -1, -1, -1, -1],
        diag_value=0.0,
        diag_bool=True,
        num_neurons=512,
        batch_size=1024,
        amplitude=1000.0,
        bias=0.05,
        motif_epochs=100,
        train_epochs=10,
        motif_loss_scale=1e5,
        motif_lr=0.01,
        train_lr=0.001,
        validation_split=0.1,
        model_dir=OUTPUT_DIR / "fmnist" / "models",
        noise_dir=OUTPUT_DIR / "fmnist" / "noise_eval",
        dataset_kwargs={
            "train_path": BASE_DIR / "datasets" / "FMNIST" / "fashion_mnist_train.npy",
            "test_path": BASE_DIR / "datasets" / "FMNIST" / "fashion_mnist_test.npy",
        },
    ),
    "tidigits": DatasetConfig(
        name="tidigits",
        num_classes=10,
        fre=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.30],
        diag_value=-0.7,
        diag_bool=True,
        num_neurons=512,
        batch_size=16,
        amplitude=1000.0,
        bias=0.05,
        motif_epochs=100,
        train_epochs=50,
        motif_loss_scale=1e5,
        motif_lr=0.01,
        train_lr=0.0001,
        validation_split=0.1,
        model_dir=OUTPUT_DIR / "tidigits" / "models",
        noise_dir=OUTPUT_DIR / "tidigits" / "noise_eval",
        dataset_kwargs={
            "packed_path": BASE_DIR / "datasets" / "TIDIGITS" / "packed_tidigits_nbands_20_nframes_20.pkl",
        },
    ),
}


def get_config(name: str) -> DatasetConfig:
    key = name.lower()
    if key not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASET_CONFIGS)}")
    return DATASET_CONFIGS[key]
