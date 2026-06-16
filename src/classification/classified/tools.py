from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import tensorflow as tf  # type: ignore
except ImportError:  # pragma: no cover - tensorflow optional
    tf = None


@dataclass
class DatasetBundle:
    train: DataLoader
    valid: Optional[DataLoader]
    test: DataLoader
    input_size: int
    sequence_length: int
    num_classes: int


class RNNClassifier(nn.Module):
    """Simple RNN classifier with an auxiliary discrete inference path."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        diag_value: float = 0.0,
        *,
        bin_mode: str = "sign",
        threshold: float = 0.0,
        n_levels: int = 8,
        diag_init: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bin_mode = bin_mode
        self.threshold = threshold
        self.n_levels = max(2, int(n_levels))
        self.diag_value = diag_value
        self.diag_init = diag_init

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.w = self.rnn.weight_hh_l0
        if self.diag_init:
            with torch.no_grad():
                diag = torch.arange(hidden_size)
                self.w[diag, diag] = self.diag_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        out, _ = self.rnn(x, h0)
        logits = self.fc(out[:, -1, :])
        return logits

    @torch.no_grad()
    def binary_forward(self, x: torch.Tensor, *, n_levels: Optional[int] = None) -> torch.Tensor:
        """Forward pass with hidden activations discretised at every timestep."""
        B, T, _ = x.shape
        W_ih = self.rnn.weight_ih_l0
        W_hh = self.rnn.weight_hh_l0
        b_ih = getattr(self.rnn, "bias_ih_l0", torch.zeros(self.hidden_size, device=x.device, dtype=x.dtype))
        b_hh = getattr(self.rnn, "bias_hh_l0", torch.zeros(self.hidden_size, device=x.device, dtype=x.dtype))
        b_total = b_ih + b_hh
        levels = self.n_levels if n_levels is None else max(2, int(n_levels))

        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(T):
            x_t = x[:, t, :]
            pre = torch.addmm(b_total, x_t, W_ih.t()) + h @ W_hh.t()
            cont = torch.tanh(pre)
            if self.bin_mode == "sign":
                h = self._quantize_symmetric_tanh(cont, levels)
            else:
                h = (cont >= self.threshold).to(dtype=x.dtype)
        logits = self.fc(h)
        return logits

    @staticmethod
    def _quantize_symmetric_tanh(y: torch.Tensor, n: int) -> torch.Tensor:
        y = torch.clamp(y, -1.0, 1.0)
        scaled = (y + 1.0) * (n - 1) / 2.0
        idx = torch.round(scaled).clamp_(0, n - 1)
        quantised = idx * (2.0 / (n - 1)) - 1.0
        return quantised


def combination(n: int, k: int) -> int:
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


class motifRegular:
    def __init__(self, fre: Sequence[float], device: torch.device, num_neuron: int, amplitude: float, bias: float) -> None:
        self.L = torch.ones([1, num_neuron], device=device)
        self.I = torch.zeros([num_neuron, num_neuron], device=device)
        self.P = torch.zeros([num_neuron, num_neuron], device=device)
        self.obs = torch.zeros([14], device=device, requires_grad=False)
        self.fre = torch.tensor(list(fre), device=device, dtype=torch.float32)
        self.sum = combination(num_neuron, 3)
        self.record_sum = 0.0
        self.amplitude = amplitude
        self.bias = bias
        self.device = device
        for i in range(num_neuron):
            self.I[i, i] = 1.0
        for i in range(num_neuron):
            for j in range(num_neuron):
                if i == j:
                    continue
                self.P[i, j] = 1.0

    def cal(self, a: torch.Tensor) -> torch.Tensor:
        m = torch.mul
        mm = torch.matmul
        a2 = a * a
        w = torch.sigmoid(self.amplitude * (a2 - self.bias * self.bias))
        w = w * self.P
        pmw = self.P - w
        w0 = pmw * pmw.T
        w1 = w * pmw.T
        w2 = pmw * w.T
        w3 = w * w.T

        q = torch.zeros([14], device=self.device)
        # q[1] = 0.5 * self.L @ (w1 * (w1 @ w0)) @ self.L.T
        # q[2] = 0.5 * self.L @ (w0 * (w1 @ w2)) @ self.L.T
        # q[3] = self.L @ (w1 * (w0 @ w2)) @ self.L.T
        # q[7] = 0.5 * self.L @ (w1 * (w1 @ w2)) @ self.L.T

        # q[4] = self.L @ (w3 * (w1 @ w0)) @ self.L.T
        # q[5] = self.L @ (w3 * (w2 @ w0)) @ self.L.T
        # q[9] = 0.5 * self.L @ (w3 * (w1 @ w2)) @ self.L.T
        # q[10] = (1.0 / 3.0) * self.L @ (w3 * (w2 @ w1)) @ self.L.T

        # q[6] = self.L @ (w3 * (w3 @ w0)) @ self.L.T
        # q[8] = 0.5 * self.L @ (w3 * (w1 @ w2)) @ self.L.T
        # q[11] = self.L @ (w3 * (w3 @ w2)) @ self.L.T
        # q[12] = self.L @ (w3 * (w3 @ w2)) @ self.L.T
        # q[13] = (1.0 / 6.0) * self.L @ (w3 * (w3 @ w3)) @ self.L.T
        q[1] = 0.5 * self.L @ (w1 * (w1 @ w0)) @ self.L.T
        q[2] = 0.5 * self.L @ (w0 * (w1 @ w2)) @ self.L.T
        q[3] = self.L @ (w1 * (w0 @ w2)) @ self.L.T
        q[4] = 0.5 * self.L @ (w1 * (w1 @ w2)) @ self.L.T

        q[5] = self.L @ (w3 * (w1 @ w0)) @ self.L.T
        q[6] = self.L @ (w3 * (w2 @ w0)) @ self.L.T
        q[7] = 0.5 * self.L @ (w3 * (w1 @ w2)) @ self.L.T
        q[8] = (1.0 / 3.0) * self.L @ (w3 * (w2 @ w1)) @ self.L.T

        q[9] = self.L @ (w3 * (w3 @ w0)) @ self.L.T
        q[10] = 0.5 * self.L @ (w3 * (w1 @ w2)) @ self.L.T
        q[11] = self.L @ (w3 * (w3 @ w2)) @ self.L.T
        q[12] = self.L @ (w3 * (w3 @ w2)) @ self.L.T
        q[13] = (1.0 / 6.0) * self.L @ (w3 * (w3 @ w3)) @ self.L.T

        r = torch.zeros([1], device=self.device)
        for i in range(13):
            if self.fre[i] >= 0:
                r += (q[i + 1] / self.sum - self.fre[i]) ** 2
        with torch.no_grad():
            for i in range(13):
                self.obs[i + 1] = q[i + 1] / self.sum
        return r.squeeze(0)


def build_diag_mask(size: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones(size, size, device=device)
    mask.fill_diagonal_(0.0)
    return mask


def _make_dataloader(tensors: Tuple[np.ndarray, np.ndarray], batch_size: int, *, shuffle: bool, drop_last: bool) -> DataLoader:
    data, labels = tensors
    ds = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def load_dataset(name: str, batch_size: int, *, validation_split: float, dataset_kwargs: Optional[dict] = None) -> DatasetBundle:
    key = name.lower()
    dataset_kwargs = dataset_kwargs or {}
    if key == "smnist":
        if tf is None:
            raise ImportError("tensorflow is required to load the SMNIST dataset")
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
        train_x = train_x.astype(np.float32) / 255.0
        test_x = test_x.astype(np.float32) / 255.0
        split = int((1.0 - validation_split) * train_x.shape[0])
        valid_x = train_x[split:]
        valid_y = train_y[split:]
        train_x = train_x[:split]
        train_y = train_y[:split]
        num_classes = int(np.max(np.concatenate([train_y, valid_y, test_y]))) + 1
        train_loader = _make_dataloader((train_x, train_y), batch_size, shuffle=True, drop_last=True)
        valid_loader = _make_dataloader((valid_x, valid_y), batch_size, shuffle=False, drop_last=False)
        test_loader = _make_dataloader((test_x, test_y), batch_size, shuffle=False, drop_last=False)
        return DatasetBundle(train_loader, valid_loader, test_loader, input_size=28, sequence_length=28, num_classes=num_classes)

    if key == "fmnist":
        train_path = Path(dataset_kwargs.get("train_path", ""))
        test_path = Path(dataset_kwargs.get("test_path", ""))
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError("Fashion-MNIST numpy files not found. Check config dataset_kwargs.")
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        train_y = train_data[:, 0].astype(np.int64)
        test_y = test_data[:, 0].astype(np.int64)
        train_x = train_data[:, 1:].astype(np.float32)
        test_x = test_data[:, 1:].astype(np.float32)

        # Per-sample normalization (zero mean, unit variance)
        for i in range(train_x.shape[0]):
            mean = np.mean(train_x[i, :])
            std = np.std(train_x[i, :])
            if std != 0:
                train_x[i, :] = (train_x[i, :] - mean) / std
        for i in range(test_x.shape[0]):
            mean = np.mean(test_x[i, :])
            std = np.std(test_x[i, :])
            if std != 0:
                test_x[i, :] = (test_x[i, :] - mean) / std

        train_x = train_x.reshape(-1, 28, 28)
        test_x = test_x.reshape(-1, 28, 28)
        split = int((1.0 - validation_split) * train_x.shape[0])
        valid_x = train_x[split:]
        valid_y = train_y[split:]
        train_x = train_x[:split]
        train_y = train_y[:split]
        num_classes = int(np.max(np.concatenate([train_y, valid_y, test_y]))) + 1
        train_loader = _make_dataloader((train_x, train_y), batch_size, shuffle=True, drop_last=True)
        valid_loader = _make_dataloader((valid_x, valid_y), batch_size, shuffle=False, drop_last=False)
        test_loader = _make_dataloader((test_x, test_y), batch_size, shuffle=False, drop_last=False)
        return DatasetBundle(train_loader, valid_loader, test_loader, input_size=28, sequence_length=28, num_classes=num_classes)

    if key == "tidigits":
        packed_path = Path(dataset_kwargs.get("packed_path", ""))
        if not packed_path.is_file():
            raise FileNotFoundError("Packed TIDigits dataset pickle not found. Check config dataset_kwargs.")
        with open(packed_path, "rb") as fh:
            data = pickle.load(fh)
        train_x = np.asarray(data[0][0], dtype=np.float32).reshape(-1, 20, 20)
        train_y = np.asarray(data[0][1], dtype=np.int64)
        test_x = np.asarray(data[2][0], dtype=np.float32).reshape(-1, 20, 20)
        test_y = np.asarray(data[2][1], dtype=np.int64)

        # Per-sample normalization (zero mean, unit variance)
        for i in range(train_x.shape[0]):
            mean = np.mean(train_x[i])
            std = np.std(train_x[i])
            if std != 0:
                train_x[i] = (train_x[i] - mean) / std
        for i in range(test_x.shape[0]):
            mean = np.mean(test_x[i])
            std = np.std(test_x[i])
            if std != 0:
                test_x[i] = (test_x[i] - mean) / std

        valid_loader: Optional[DataLoader] = None
        valid_labels: Optional[np.ndarray] = None
        if len(data) > 1 and len(data[1][0]) > 0:
            valid_x = np.asarray(data[1][0], dtype=np.float32).reshape(-1, 20, 20)
            valid_labels = np.asarray(data[1][1], dtype=np.int64)
            for i in range(valid_x.shape[0]):
                mean = np.mean(valid_x[i])
                std = np.std(valid_x[i])
                if std != 0:
                    valid_x[i] = (valid_x[i] - mean) / std
            valid_loader = _make_dataloader((valid_x, valid_labels), batch_size, shuffle=False, drop_last=False)
        elif validation_split > 0.0:
            split_idx = int((1.0 - validation_split) * train_x.shape[0])
            split_idx = max(1, min(train_x.shape[0] - 1, split_idx))
            valid_x = train_x[split_idx:]
            valid_labels = train_y[split_idx:]
            train_x = train_x[:split_idx]
            train_y = train_y[:split_idx]
            if valid_x.size > 0:
                for i in range(valid_x.shape[0]):
                    mean = np.mean(valid_x[i])
                    std = np.std(valid_x[i])
                    if std != 0:
                        valid_x[i] = (valid_x[i] - mean) / std
                valid_loader = _make_dataloader((valid_x, valid_labels), batch_size, shuffle=False, drop_last=False)

        labels_for_classes = [train_y, test_y]
        if valid_labels is not None and valid_labels.size > 0:
            labels_for_classes.append(valid_labels)
        num_classes = int(np.max(np.concatenate(labels_for_classes))) + 1

        train_loader = _make_dataloader((train_x, train_y), batch_size, shuffle=True, drop_last=False)
        test_loader = _make_dataloader((test_x, test_y), batch_size, shuffle=False, drop_last=False)
        return DatasetBundle(train_loader, valid_loader, test_loader, input_size=20, sequence_length=20, num_classes=num_classes)

    raise KeyError(f"Unknown dataset '{name}'.")


def motif_initialisation(
    model: RNNClassifier,
    motif_reg: motifRegular,
    epochs: int,
    optimiser: torch.optim.Optimizer,
    *,
    scale: float,
    print_every: int = 20,
) -> List[float]:
    history: List[float] = []
    if epochs <= 0:
        return history
    for epoch in range(epochs):
        loss_val = motif_reg.cal(model.w)
        loss = scale * loss_val
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        history.append(loss_val.item())
        if epoch % print_every == 0:
            print(f"[Motif] epoch={epoch:03d} loss={loss_val.item():.6f}")
    return history


def train_task(
    model: RNNClassifier,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader],
    epochs: int,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    diag_mask: Optional[torch.Tensor] = None,
) -> List[dict]:
    history: List[dict] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimiser.zero_grad()
            loss.backward()
            if diag_mask is not None and model.w.grad is not None:
                model.w.grad *= diag_mask
            optimiser.step()
            total_loss += loss.item()
        log = {"epoch": epoch + 1, "train_loss": total_loss}
        if valid_loader is not None:
            cont_acc = evaluate(model, valid_loader, device, binary=False)
            bin_acc = evaluate(model, valid_loader, device, binary=True)
            log.update({"valid_cont": cont_acc, "valid_bin": bin_acc})
        history.append(log)
        msg = f"[Train] epoch={epoch+1:03d} loss={total_loss:.4f}"
        if valid_loader is not None:
            msg += f" | val_cont={log['valid_cont']:.2f}% val_bin={log['valid_bin']:.2f}%"
        print(msg)
    return history


def evaluate(
    model: RNNClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    binary: bool = False,
    noise_variance: float = 0.0,
    discrete_levels: Optional[int] = None,
) -> float:
    model.eval()
    total = 0
    correct = 0
    sigma = float(np.sqrt(noise_variance)) if noise_variance > 0 else 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if sigma > 0:
                inputs = inputs + torch.randn_like(inputs) * sigma
            if binary:
                outputs = model.binary_forward(inputs, n_levels=discrete_levels)
            else:
                outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def noise_sweep(
    model: RNNClassifier,
    loader: DataLoader,
    device: torch.device,
    variances: Sequence[float],
    *,
    discrete_levels: Optional[int] = None,
) -> np.ndarray:
    rows: List[Tuple[float, float, float]] = []
    for v in variances:
        cont = evaluate(model, loader, device, binary=False, noise_variance=v)
        disc = evaluate(model, loader, device, binary=True, noise_variance=v, discrete_levels=discrete_levels)
        rows.append((float(v), cont, disc))
        print(f"noise_var={v:.6f} -> cont={cont:.2f}% disc={disc:.2f}%")
    return np.asarray(rows, dtype=np.float32)


def save_model(model: RNNClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "bin_mode": model.bin_mode,
        "threshold": model.threshold,
        "n_levels": model.n_levels,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
    }
    torch.save(payload, path)


def load_trained_model(
    path: Path,
    *,
    input_size: int,
    output_size: int,
    diag_value: float,
    bin_mode: str,
    threshold: float,
    n_levels: int,
    device: torch.device,
) -> RNNClassifier:
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    if isinstance(payload, nn.Module):
        model = payload
        model.to(device)
        model.eval()
        return model
    if isinstance(payload, dict) and "state_dict" in payload:
        hidden_size = payload.get("hidden_size") or payload["state_dict"]["rnn.weight_hh_l0"].shape[0]
        num_layers = payload.get("num_layers", 1)
        model = RNNClassifier(
            input_size,
            hidden_size,
            output_size,
            num_layers=num_layers,
            diag_value=diag_value,
            bin_mode=payload.get("bin_mode", bin_mode),
            threshold=payload.get("threshold", threshold),
            n_levels=payload.get("n_levels", n_levels),
            diag_init=False,
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        return model
    # fallback: pickle with entire module
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Unsupported checkpoint type: {type(model)}")
    model.to(device)
    model.eval()
    return model
