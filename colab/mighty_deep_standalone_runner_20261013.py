"""Self-contained Colab MightyNet bootstrap runner.
Usage: python mighty_standalone_runner_20261013.py --dataset data.xlsx --networks 1000
Supports CSV/XLSX with binary target labels and finite numeric features.
Uses 1.6*n replacement draws, disjoint architecture/leaf patients, per-model
normalization, early stopping and Platt calibration; metrics use OOB rows only.
No RPCA preprocessing is applied. Model implementation copied from mighty_net.py.
"""

import argparse
import gc
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import stack_module_state, functional_call, vmap
from sklearn.metrics import roc_auc_score, roc_curve

VERBOSE = 100


class MightyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.0, num_hidden_layers=2):
        super(MightyNet, self).__init__()
        if num_hidden_layers not in (1, 2):
            raise ValueError("num_hidden_layers must be 1 or 2.")
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = (
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)) if num_hidden_layers == 2 else None
        )
        self.fc3 = nn.Linear(max(1, hidden_dim // 2) if num_hidden_layers == 2 else hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        if self.fc2 is not None:
            x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# -----------------------------
# Data splits
# -----------------------------


def _feature_weights(X, feature_frac):
    if not 0 < feature_frac <= 1:
        raise ValueError("feature_frac must be in (0, 1].")
    if feature_frac == 1.0:
        return None  # No feature sampling: avoid an unnecessary PCA fit.
    from sklearn.decomposition import PCA

    values = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    pca = PCA().fit(values)
    weights = np.sum(np.abs(pca.components_) * pca.explained_variance_[:, None], axis=0)
    weights = np.clip(weights, 1e-6, None)
    return weights / weights.sum()


def _sample_features(m, feature_frac, weights, rng):
    count = max(1, int(round(m * feature_frac)))
    if count == m:
        return np.arange(m)
    return rng.choice(m, size=count, replace=False, p=weights)


def make_might_splits(
    X,
    y=None,
    num_models=100,
    arch_frac=0.4,
    leaf_frac=0.4,
    test_frac=0.2,
    feature_frac=1.0,
    seed=0,
):
    """
    Exact non-overlapping 40/40/20 splits per model for the samples.
    For features, a random subset of size feature_frac from range(m) is selected for each model.

    Note: If you use bootstrap + OOB, the OOB fraction is about 36.8%,
    not 20%. This function implements the stated 40/40/20 design.
    """
    if abs(arch_frac + leaf_frac + test_frac - 1.0) >= 1e-8:
        raise ValueError("Split fractions must sum to one.")
    if not 0 < feature_frac <= 1:
        raise ValueError("feature_frac must be in (0, 1].")

    n, m = X.shape
    feature_weights = _feature_weights(X, feature_frac)

    rng = np.random.default_rng(seed)
    if y is None:
        y = np.zeros(n, dtype=int)
    y = (
        y.detach().cpu().numpy().reshape(-1)
        if isinstance(y, torch.Tensor)
        else np.asarray(y).reshape(-1)
    )
    if len(y) != n:
        raise ValueError(f"Expected y to have length {n}, got {len(y)}.")

    def apportion(target_total, weights):
        weights = np.asarray(weights, dtype=float)

        if target_total <= 0:
            return np.zeros_like(weights, dtype=int)

        weight_sum = weights.sum()
        if weight_sum <= 0:
            return np.zeros_like(weights, dtype=int)

        desired = weights / weight_sum * target_total
        base = np.floor(desired).astype(int)
        remainder = int(target_total - base.sum())

        if remainder > 0:
            fractional = desired - base
            tie_break = rng.random(len(weights)) * 1e-12
            order = np.argsort(-(fractional + tie_break))
            base[order[:remainder]] += 1

        return base

    arch_indices = []
    leaf_indices = []
    test_indices = []
    feature_indices = []

    labels, class_counts = np.unique(y, return_counts=True)
    class_indices = {label: np.where(y == label)[0] for label in labels}

    n_test = int(round(n * test_frac))
    n_arch = int(round(n * arch_frac))
    n_leaf = n - n_test - n_arch

    for _ in range(num_models):
        shuffled_by_label = {
            label: rng.permutation(indices) for label, indices in class_indices.items()
        }

        test_counts = apportion(n_test, class_counts)
        train_counts = class_counts - test_counts
        arch_counts = apportion(n_arch, train_counts)

        arch_parts = []
        leaf_parts = []
        test_parts = []

        for label_idx, label in enumerate(labels):
            shuffled = shuffled_by_label[label]

            n_test_class = int(test_counts[label_idx])
            n_arch_class = int(arch_counts[label_idx])

            test = shuffled[:n_test_class]
            train = shuffled[n_test_class:]

            arch = train[:n_arch_class]
            leaf = train[n_arch_class:]

            test_parts.append(test)
            arch_parts.append(arch)
            leaf_parts.append(leaf)

        arch = rng.permutation(np.concatenate(arch_parts))
        leaf = rng.permutation(np.concatenate(leaf_parts))
        test = rng.permutation(np.concatenate(test_parts))

        if len(arch) != n_arch or len(leaf) != n_leaf or len(test) != n - n_arch - n_leaf:
            raise RuntimeError("Stratified split produced unexpected split sizes.")

        arch_indices.append(arch)
        leaf_indices.append(leaf)
        test_indices.append(test)
        # select features based on signal weights
        feature_indices.append(_sample_features(m, feature_frac, feature_weights, rng))
    return arch_indices, leaf_indices, test_indices, feature_indices


def make_bootstrap_might_splits(X, y=None, num_models=100, feature_frac=1.0, seed=0):
    """
    Bootstrap-style split.

    Unique bootstrap samples are split into architecture/leaf sets.
    If y is provided, the unique bootstrap pool is split in a stratified way.
    Out-of-bag patients become test patients.

    Expected proportions are roughly:
      architecture: 31.6%
      leaf:         31.6%
      OOB test:     36.8%

    This mirrors RF-style OOB logic more closely than exact 40/40/20.
    """
    rng = np.random.default_rng(seed)
    stratify = y is not None
    n, m = X.shape
    feature_weights = _feature_weights(X, feature_frac)
    if y is None:
        y = np.zeros(n, dtype=int)
    y = (
        y.detach().cpu().numpy().reshape(-1)
        if isinstance(y, torch.Tensor)
        else np.asarray(y).reshape(-1)
    )
    if len(y) != n:
        raise ValueError(f"Expected y to have length {n}, got {len(y)}.")

    def apportion(target_total, weights):
        weights = np.asarray(weights, dtype=float)

        if target_total <= 0:
            return np.zeros_like(weights, dtype=int)

        weight_sum = weights.sum()
        if weight_sum <= 0:
            return np.zeros_like(weights, dtype=int)

        desired = weights / weight_sum * target_total
        base = np.floor(desired).astype(int)
        remainder = int(target_total - base.sum())

        if remainder > 0:
            fractional = desired - base
            tie_break = rng.random(len(weights)) * 1e-12
            order = np.argsort(-(fractional + tie_break))
            base[order[:remainder]] += 1

        return base

    all_idx = np.arange(n)
    arch_indices = []
    leaf_indices = []
    test_indices = []
    feature_indices = []

    labels, class_counts = np.unique(y, return_counts=True)
    class_indices = {label: np.where(y == label)[0] for label in labels}

    for _ in range(num_models):
        boot = rng.choice(n, size=n, replace=True)
        unique_boot = np.unique(boot)

        if len(unique_boot) == 0:
            raise RuntimeError("Bootstrap sample produced no unique indices.")

        if not stratify:
            rng.shuffle(unique_boot)
            mid = len(unique_boot) // 2

            arch_indices.append(unique_boot[:mid])
            leaf_indices.append(unique_boot[mid:])
        else:
            unique_by_label = {
                label: np.intersect1d(class_indices[label], unique_boot, assume_unique=False)
                for label in labels
            }

            unique_counts = np.array([len(unique_by_label[label]) for label in labels])
            arch_counts = apportion(len(unique_boot) // 2, unique_counts)

            arch_parts = []
            leaf_parts = []

            for label_idx, label in enumerate(labels):
                shuffled = rng.permutation(unique_by_label[label])
                n_arch_class = int(arch_counts[label_idx])

                arch_parts.append(shuffled[:n_arch_class])
                leaf_parts.append(shuffled[n_arch_class:])

            arch_indices.append(rng.permutation(np.concatenate(arch_parts)))
            leaf_indices.append(rng.permutation(np.concatenate(leaf_parts)))

        test_indices.append(np.setdiff1d(all_idx, unique_boot))
        feature_indices.append(_sample_features(m, feature_frac, feature_weights, rng))

    return arch_indices, leaf_indices, test_indices, feature_indices


# -----------------------------
# Per-model batch sampler
# -----------------------------


def _pack_indices(index_sets, device):
    index_sets = [torch.as_tensor(i, device=device, dtype=torch.long) for i in index_sets]
    if not index_sets or any(len(i) == 0 for i in index_sets):
        raise ValueError("Every model needs a nonempty index set.")
    lengths = torch.tensor([len(i) for i in index_sets], device=device)
    padded = torch.zeros(
        (len(index_sets), int(lengths.max().item())), device=device, dtype=torch.long
    )
    for model_idx, indices in enumerate(index_sets):
        padded[model_idx, : len(indices)] = indices
    return padded, lengths


def sample_model_batches(X, y, index_sets, feature_sets, batch_size, device):
    """
    Returns:
      xb: [num_models, batch_size, input_dim]
      yb: [num_models, batch_size, 1]

    Vectorized GPU sampling without host-device transfers.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, device=device, dtype=torch.float32)
    elif X.device != torch.device(device):
        X = X.to(device)

    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y, device=device, dtype=torch.float32)
    elif y.device != torch.device(device):
        y = y.to(device)

    if y.ndim == 1:
        y = y.unsqueeze(-1)

    padded, lengths = (
        index_sets if isinstance(index_sets, tuple) else _pack_indices(index_sets, device)
    )
    features = (
        feature_sets
        if isinstance(feature_sets, torch.Tensor)
        else torch.stack(
            [torch.as_tensor(f, device=device, dtype=torch.long) for f in feature_sets]
        )
    )
    positions = (torch.rand((len(lengths), batch_size), device=device) * lengths[:, None]).long()
    chosen = padded.gather(1, positions)
    return X[chosen.unsqueeze(-1), features[:, None, :]].float(), y[chosen].float()


# -----------------------------
# Vectorized ensemble wrapper
# -----------------------------


class DeepMIGHT:
    def __init__(
        self,
        input_dim,
        num_models=100,
        feature_indices=None,
        hidden_dim=64,
        dropout=0.0,
        device=None,
        num_hidden_layers=2,
        normalize_inputs=False,
    ):
        self.input_dim = input_dim
        if isinstance(num_models, list):
            num_models = len(num_models)

        self.num_models = num_models
        self.normalize_inputs = normalize_inputs
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if feature_indices is None:
            feature_indices = [np.arange(input_dim) for _ in range(self.num_models)]
        self.feature_indices = [np.asarray(f, dtype=np.int64) for f in feature_indices]
        if len(self.feature_indices) != self.num_models or any(
            len(f) != input_dim for f in self.feature_indices
        ):
            raise ValueError("Provide one feature set per model, each of input_dim length.")

        self.base_model = MightyNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_hidden_layers=num_hidden_layers,
        ).to(self.device)

        models = [
            MightyNet(input_dim, hidden_dim, dropout, num_hidden_layers).to(self.device)
            for _ in range(self.num_models)
        ]
        self.params, self.buffers = stack_module_state(models)

        self.params = {k: v.to(self.device).requires_grad_(True) for k, v in self.params.items()}
        self.buffers = {k: v.to(self.device) for k, v in self.buffers.items()}

        def single_forward(params, buffers, x):
            return functional_call(self.base_model, (params, buffers), x)

        self.single_forward = single_forward
        # Use different randomness per mapped instance so stochastic ops (Dropout) work under vmap
        self.vmap_forward = vmap(single_forward, in_dims=(0, 0, 0), randomness="different")

    def fit_networks(
        self,
        X,
        y,
        arch_indices,
        epochs=50,
        batch_size=128,
        lr=1e-3,
        weight_decay=0.0,
        verbose=True,
        tol=1e-2,
        validation_indices=None,
        patience=5,
        validation_interval=5,
    ):
        """Fit models with independent checkpoints and patience on disjoint leaf rows.

        Input scaling is estimated separately for each model from its architecture
        rows, including bootstrap multiplicities. Leaf and OOB rows do not affect
        the scaling statistics.
        """
        X = (
            X.to(self.device).float()
            if isinstance(X, torch.Tensor)
            else torch.as_tensor(X, device=self.device, dtype=torch.float32)
        )
        y = (
            y.to(self.device).float()
            if isinstance(y, torch.Tensor)
            else torch.as_tensor(y, device=self.device, dtype=torch.float32)
        )
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        arch_indices_gpu = _pack_indices(arch_indices, self.device)
        feature_indices_gpu = torch.stack(
            [
                torch.as_tensor(idx, device=self.device, dtype=torch.long)
                for idx in self.feature_indices
            ]
        )
        if self.normalize_inputs:
            means, scales = [], []
            for rows, features in zip(arch_indices, feature_indices_gpu):
                rows = torch.as_tensor(rows, device=self.device, dtype=torch.long)
                training_values = X[rows][:, features]
                means.append(training_values.mean(dim=0))
                scales.append(training_values.std(dim=0, unbiased=False).clamp_min(1e-8))
            self.feature_mean = torch.stack(means)
            self.feature_scale = torch.stack(scales)
        if validation_indices is not None:
            validation_gpu = _pack_indices(validation_indices, self.device)
            valid_rows, valid_lengths = validation_gpu
            valid_mask = (
                torch.arange(valid_rows.shape[1], device=self.device)[None, :]
                < valid_lengths[:, None]
            )
            if patience < 1 or validation_interval < 1:
                raise ValueError("patience and validation_interval must be positive.")
            if len(validation_indices) != self.num_models:
                raise ValueError("Provide one validation index set per model.")
            best_loss = torch.full((self.num_models,), float("inf"), device=self.device)
            best_params = {k: v.detach().clone() for k, v in self.params.items()}
            stale_checks = torch.zeros(self.num_models, dtype=torch.long, device=self.device)
            active = torch.ones(self.num_models, dtype=torch.bool, device=self.device)
            best_epoch = torch.zeros(self.num_models, dtype=torch.long, device=self.device)
            stopped_epoch = torch.zeros_like(best_epoch)
        optimizer = torch.optim.Adam(
            self.params.values(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.base_model.train()
        for epoch in range(epochs):
            xb, yb = sample_model_batches(
                X=X,
                y=y,
                index_sets=arch_indices_gpu,
                feature_sets=feature_indices_gpu,
                batch_size=batch_size,
                device=self.device,
            )
            if self.normalize_inputs:
                xb = ((xb - self.feature_mean[:, None, :]) / self.feature_scale[:, None, :]).clamp(
                    -5, 5
                )

            optimizer.zero_grad()

            logits = self.vmap_forward(self.params, self.buffers, xb)
            model_losses = F.binary_cross_entropy_with_logits(logits, yb, reduction="none").mean(
                dim=(1, 2)
            )
            if validation_indices is not None:
                loss = (model_losses * active).sum()
            else:
                loss = model_losses.sum()

            loss.backward()

            optimizer.step()
            if validation_indices is not None:
                # Adam momentum can move stopped slices even with zero gradients.
                # Restore their own checkpoints after every optimizer step.
                with torch.no_grad():
                    for name, value in self.params.items():
                        mask = active.reshape((-1,) + (1,) * (value.ndim - 1))
                        value.copy_(torch.where(mask, value, best_params[name]))

            if verbose and (epoch == 0 or (epoch + 1) % VERBOSE == 0):
                print(f"network epoch {epoch + 1:04d} | loss={loss.item():.4f}")
            if validation_indices is None and loss.item() < tol:
                break
            if validation_indices is not None and (
                (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
            ):
                self.base_model.eval()
                with torch.no_grad():
                    valid_x = X[valid_rows.unsqueeze(-1), feature_indices_gpu[:, None, :]]
                    if self.normalize_inputs:
                        valid_x = (
                            (valid_x - self.feature_mean[:, None, :])
                            / self.feature_scale[:, None, :]
                        ).clamp(-5, 5)
                    valid_y = y[valid_rows]
                    valid_logits = self.vmap_forward(self.params, self.buffers, valid_x)
                    valid_losses = F.binary_cross_entropy_with_logits(
                        valid_logits, valid_y, reduction="none"
                    )
                    valid_loss = (
                        valid_losses.squeeze(-1).masked_fill(~valid_mask, 0).sum(dim=1)
                        / valid_lengths
                    )
                self.base_model.train()
                improved = active & (valid_loss < best_loss - 1e-4)
                with torch.no_grad():
                    best_loss = torch.where(improved, valid_loss, best_loss)
                    best_epoch[improved] = epoch + 1
                    for name, value in self.params.items():
                        mask = improved.reshape((-1,) + (1,) * (value.ndim - 1))
                        best_params[name].copy_(
                            torch.where(mask, value.detach(), best_params[name])
                        )
                stale_checks = torch.where(
                    improved,
                    torch.zeros_like(stale_checks),
                    stale_checks + active.long(),
                )
                newly_stopped = active & (stale_checks >= patience)
                stopped_epoch[newly_stopped] = epoch + 1
                active &= ~newly_stopped
                if not active.any().item():
                    break
        if validation_indices is not None:
            stopped_epoch[active] = epochs
            self.best_validation_loss_ = best_loss.detach().cpu()
            self.best_epoch_ = best_epoch.detach().cpu()
            self.stopped_epoch_ = stopped_epoch.detach().cpu()
            with torch.no_grad():
                for name, value in self.params.items():
                    value.copy_(best_params[name])
        self.base_model.eval()

    @torch.no_grad()
    def raw_logits_for_model(self, model_idx, X):
        p = {k: v[model_idx] for k, v in self.params.items()}
        b = {k: v[model_idx] for k, v in self.buffers.items()}

        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        features = torch.as_tensor(
            self.feature_indices[model_idx], device=self.device, dtype=torch.long
        )
        X = X[:, features]
        if self.normalize_inputs:
            X = ((X - self.feature_mean[model_idx]) / self.feature_scale[model_idx]).clamp(-5, 5)
        return self.single_forward(p, b, X).squeeze(-1)

    @torch.no_grad()
    def raw_logits_for_test(self, model_idx, test_indices, feature_indices, X):
        p = {k: v[model_idx] for k, v in self.params.items()}
        b = {k: v[model_idx] for k, v in self.buffers.items()}
        # Support per-model split lists as well as direct index arrays.
        if isinstance(test_indices, (list, tuple)):
            first = test_indices[0] if len(test_indices) > 0 else None
            if first is not None and isinstance(first, (np.ndarray, list, tuple, torch.Tensor)):
                test_indices = test_indices[model_idx]

        idx = torch.as_tensor(test_indices, device=self.device, dtype=torch.long)
        if isinstance(feature_indices, (list, tuple)):
            first = feature_indices[0] if len(feature_indices) > 0 else None
            if first is not None and isinstance(first, (np.ndarray, list, tuple, torch.Tensor)):
                feature_indices = feature_indices[model_idx]

        feature_idx = torch.as_tensor(feature_indices, device=self.device, dtype=torch.long)
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        X_test = X[idx][:, feature_idx]
        if self.normalize_inputs:
            X_test = (
                (X_test - self.feature_mean[model_idx]) / self.feature_scale[model_idx]
            ).clamp(-5, 5)
        return self.single_forward(p, b, X_test).squeeze(-1)

    @torch.no_grad()
    def _leaf_logits(self, X, y, leaf_indices):
        """Cache held-out calibration logits once per model."""
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32).reshape(-1)
        logits, labels = [], []
        for model_idx, indices in enumerate(leaf_indices):
            indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            if len(indices) == 0:
                raise ValueError(f"Model {model_idx} has no calibration samples.")
            logits.append(self.raw_logits_for_model(model_idx, X[indices]))
            labels.append(y[indices])
        width = max(map(len, logits))
        values = torch.zeros((self.num_models, width), device=self.device)
        targets = torch.zeros_like(values)
        mask = torch.zeros_like(values, dtype=torch.bool)
        for model_idx, (z, target) in enumerate(zip(logits, labels)):
            values[model_idx, : len(z)] = z
            targets[model_idx, : len(z)] = target
            mask[model_idx, : len(z)] = True
        return values, targets, mask

    def fit_platt_calibrators(
        self,
        X,
        y,
        leaf_indices,
        epochs=300,
        lr=5e-2,
        batch_size=256,
        verbose=True,
        tol=1e-1,
    ):
        """Fit Platt parameters with an independent loss and stop per model."""
        values, targets, mask = self._leaf_logits(X, y, leaf_indices)
        lengths = mask.sum(dim=1)
        slope = nn.Parameter(torch.ones(self.num_models, 1, device=self.device))
        intercept = nn.Parameter(torch.zeros(self.num_models, 1, device=self.device))
        active = torch.ones(self.num_models, dtype=torch.bool, device=self.device)
        stopped_epoch = torch.zeros(self.num_models, dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam([slope, intercept], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            losses = F.binary_cross_entropy_with_logits(
                slope * values + intercept, targets, reduction="none"
            )
            model_losses = losses.masked_fill(~mask, 0).sum(dim=1) / lengths
            newly_stopped = active & (model_losses.detach() < tol)
            stopped_epoch[newly_stopped] = epoch + 1
            active &= ~newly_stopped
            if verbose and (epoch == 0 or (epoch + 1) % VERBOSE == 0):
                print(
                    f"platt epoch {epoch + 1:04d} | "
                    f"mean loss={model_losses.mean().item():.4f} | "
                    f"active models={active.sum().item()}"
                )
            if not active.any().item():
                break
            loss = (model_losses * active).sum()
            loss.backward()
            # Adam momentum can change inactive slices despite zero gradients.
            frozen_slope = slope.detach().clone()
            frozen_intercept = intercept.detach().clone()
            optimizer.step()
            with torch.no_grad():
                slope.copy_(torch.where(active[:, None], slope, frozen_slope))
                intercept.copy_(torch.where(active[:, None], intercept, frozen_intercept))
        stopped_epoch[active] = epochs
        self.platt_a = slope.detach()
        self.platt_b = intercept.detach()
        self.platt_stopped_epoch_ = stopped_epoch.detach().cpu()
        with torch.no_grad():
            final_losses = (
                F.binary_cross_entropy_with_logits(
                    slope * values + intercept, targets, reduction="none"
                )
                .masked_fill(~mask, 0)
                .sum(dim=1)
                / lengths
            )
        self.platt_loss_ = final_losses.cpu()

    def fit_lda_calibrators(
        self,
        X,
        y,
        leaf_indices,
        epochs=300,
        lr=5e-2,
        batch_size=256,
        verbose=True,
        tol=1e-1,
    ):
        """Fit one-dimensional Gaussian LDA to each model's leaf logits."""
        values, targets, mask = self._leaf_logits(X, y, leaf_indices)
        slopes, intercepts = [], []
        for model_idx in range(self.num_models):
            z = values[model_idx, mask[model_idx]]
            labels = targets[model_idx, mask[model_idx]]
            positive, negative = z[labels == 1], z[labels == 0]
            n1, n0 = len(positive), len(negative)
            prior = np.log((n1 + 0.5) / (n0 + 0.5))
            if not n1 or not n0:
                slopes.append(0.0)
                intercepts.append(prior)
                continue
            mu1, mu0 = positive.mean(), negative.mean()
            variance = (((positive - mu1) ** 2).sum() + ((negative - mu0) ** 2).sum()) / max(
                n1 + n0 - 2, 1
            )
            variance = variance.clamp_min(1e-6)
            slopes.append(((mu1 - mu0) / variance).item())
            intercepts.append((prior - (mu1.square() - mu0.square()) / (2 * variance)).item())
        self.lda_w = torch.tensor(slopes, device=self.device)
        self.lda_b = torch.tensor(intercepts, device=self.device)

    # wrapper function for fitting calibrators using the chosen method
    def fit_calibrators(
        self,
        method,
        X,
        y,
        leaf_indices,
        epochs=300,
        lr=5e-2,
        batch_size=256,
        verbose=True,
        tol=1e-1,
    ):
        if method == "platt":
            self.fit_platt_calibrators(
                X,
                y,
                leaf_indices,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                verbose=verbose,
                tol=tol,
            )
        elif method == "lda":
            self.fit_lda_calibrators(
                X,
                y,
                leaf_indices,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                verbose=verbose,
                tol=tol,
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    # wrapper for calibrated probability prediction, allowing choice of calibration method
    def predict_calibrated_proba(self, model_idx, X, sigmoid=True, method="platt"):
        """
        Predicts calibrated probabilities for a single model.

        Args:
          model_idx: Index of the model.
          X: Input features.
          sigmoid: Whether to apply the sigmoid function to the calibrated logits.
          method: Calibration method, either "platt" or "lda".

        Returns:
          Calibrated probabilities for the specified model.
        """
        if method not in ("platt", "lda"):
            raise ValueError(f"Unknown calibration method: {method}")
        if method == "lda" and not hasattr(self, "lda_w"):
            raise ValueError("LDA calibration has not been fitted yet.")
        if method == "platt" and not hasattr(self, "platt_a"):
            raise ValueError("Platt calibration has not been fitted yet.")
        if method == "platt" and hasattr(self, "platt_a"):
            return self.predict_model_proba(model_idx, X, sigmoid=sigmoid, method="platt")
        if method == "lda" and hasattr(self, "lda_w"):
            return self.predict_model_proba(model_idx, X, sigmoid=sigmoid, method="lda")

    @torch.no_grad()
    def predict_model_proba(self, model_idx, X, sigmoid=True, method="platt"):
        raw = self.raw_logits_for_model(model_idx, X)

        if method == "platt":
            calibrated_logit = (
                self.platt_a[model_idx].squeeze() * raw + self.platt_b[model_idx].squeeze()
            )
        elif method == "lda":
            calibrated_logit = (
                self.lda_w[model_idx].squeeze() * raw + self.lda_b[model_idx].squeeze()
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        if sigmoid:
            return torch.sigmoid(calibrated_logit)
        else:
            return calibrated_logit

    @torch.no_grad()
    def predict_test_proba(
        self, model_idx, test_indices, feature_indices, X, method="platt", sigmoid=True
    ):
        raw_test = self.raw_logits_for_test(model_idx, test_indices, feature_indices, X)

        if method == "platt":
            calibrated_logit = (
                self.platt_a[model_idx].squeeze() * raw_test + self.platt_b[model_idx].squeeze()
            )
        elif method == "lda":
            calibrated_logit = (
                self.lda_w[model_idx].squeeze() * raw_test + self.lda_b[model_idx].squeeze()
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        if sigmoid:
            return torch.sigmoid(calibrated_logit)
        else:
            return calibrated_logit

    @torch.no_grad()
    def predict_oob_matrix(self, X, test_indices, feature_indices, calibration="platt"):
        """
        Returns:
          probs: [num_models, n_samples], NaN where sample is not OOB/test
          counts: [n_samples], number of models that predicted each sample
        """
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        n = X.shape[0]

        probs = torch.full(
            (self.num_models, n),
            float("nan"),
            device=self.device,
        )

        for m in range(self.num_models):
            model_test_indices = (
                test_indices[m] if isinstance(test_indices, (list, tuple)) else test_indices
            )
            model_feature_indices = (
                feature_indices[m]
                if isinstance(feature_indices, (list, tuple))
                else feature_indices
            )
            pm = self.predict_test_proba(
                m, model_test_indices, model_feature_indices, X, method=calibration
            )
            rows = torch.as_tensor(model_test_indices, device=self.device, dtype=torch.long)
            probs[m, rows] = pm

        counts = torch.isfinite(probs).sum(dim=0)

        return probs, counts

    @torch.no_grad()
    def predict_oob_mean(self, X, test_indices, feature_indices, calibration="platt"):
        """
        Aggregates each patient's held-out predictions across all models where
        that patient was in the model's test/OOB set.
        """
        probs, counts = self.predict_oob_matrix(
            X, test_indices, feature_indices, calibration=calibration
        )
        summed = torch.nan_to_num(probs, nan=0.0).sum(dim=0)
        mean = summed / counts.clamp_min(1)

        mean[counts == 0] = float("nan")

        return mean.cpu(), counts.cpu()


# -----------------------------
def bootstrap_indices(n_samples, *, n_est=100, sample_size=1.6, random_state=0):
    """Match the TabICL replacement draws and disjoint out-of-bag indices."""
    if n_samples < 2 or n_est < 1 or not np.isfinite(sample_size) or sample_size <= 0:
        raise ValueError("Bootstrap sizes and estimator count must be positive.")
    rng = np.random.default_rng(random_state)
    context_size = int(np.ceil(sample_size * n_samples))
    for _ in range(n_est):
        draw = rng.integers(0, n_samples, size=context_size)
        in_bag = np.zeros(n_samples, dtype=bool)
        in_bag[draw] = True
        yield draw, np.flatnonzero(~in_bag)


def split_bootstrap_context(draw, y, rng):
    """Partition unique in-bag patients by class for disjoint network and leaf sets.

    Repeated bootstrap draws remain in the network training set. Calibration
    uses each assigned leaf patient once.
    """
    unique = np.unique(draw)
    architecture, leaf = [], []
    for label in (0, 1):
        members = rng.permutation(unique[y[unique] == label])
        if len(members) < 2:
            raise ValueError("A bootstrap context needs at least two patients per class.")
        midpoint = len(members) // 2
        architecture.append(members[:midpoint])
        leaf.append(members[midpoint:])
    architecture = np.concatenate(architecture)
    leaf = np.concatenate(leaf)
    return draw[np.isin(draw, architecture)], rng.permutation(leaf)


def metwork_bootstrap(
    X,
    y,
    *,
    random_state=0,
    n_est=100,
    sample_size=1.6,
    model_batch=10,
    epochs=100,
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-2,
    hidden_dim=32,
    dropout=0.3,
    num_hidden_layers=2,
    patience=5,
    calibration_epochs=150,
    calibration="platt",
    device=None,
):
    """Train compact MightyNet ensembles and return mean OOB probabilities.

    The defaults were selected with small repeated OOB comparisons on this
    179-by-5340 RPCA residual matrix. Each model scales only its own training
    rows and uses its disjoint leaf rows for stopping and Platt calibration.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("X must be a 2D matrix with matching binary labels.")
    if model_batch < 1 or n_est < 1:
        raise ValueError("model_batch and n_est must be positive.")
    if calibration not in ("platt", "lda"):
        raise ValueError("calibration must be 'platt' or 'lda'.")
    torch.manual_seed(random_state)
    rng = np.random.default_rng(random_state)
    draws = []
    # Reject rare draws lacking two unique in-bag patients in either class.
    # Bound retries so small/imbalanced datasets cannot hang indefinitely.
    draw_rng = np.random.default_rng(random_state)
    attempts = 0
    while len(draws) < n_est:
        attempts += 1
        if attempts > max(1000, n_est * 100):
            raise ValueError("Too few valid bootstrap contexts; more samples per class are needed.")
        draw = draw_rng.integers(0, len(y), size=int(np.ceil(sample_size * len(y))))
        unique = np.unique(draw)
        if any(np.sum(y[unique] == label) < 2 for label in (0, 1)):
            continue
        architecture, leaf = split_bootstrap_context(draw, y, rng)
        test = np.setdiff1d(np.arange(len(y)), unique)
        draws.append((architecture, leaf, test))

    sums = np.zeros(len(y), dtype=np.float64)
    counts = np.zeros(len(y), dtype=np.int32)
    all_features = np.arange(X.shape[1], dtype=np.int64)
    for start in range(0, n_est, model_batch):
        group = draws[start : start + model_batch]
        features = [all_features for _ in group]
        model = DeepMIGHT(
            input_dim=X.shape[1],
            num_models=len(group),
            feature_indices=features,
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_hidden_layers=num_hidden_layers,
            normalize_inputs=True,
        )
        model.fit_networks(
            X,
            y,
            [entry[0] for entry in group],
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            weight_decay=weight_decay,
            validation_indices=[entry[1] for entry in group],
            patience=patience,
            verbose=False,
        )
        model.fit_calibrators(
            calibration,
            X,
            y,
            [entry[1] for entry in group],
            epochs=calibration_epochs,
            verbose=False,
        )
        for local_idx, (_, _, test) in enumerate(group):
            if not len(test):
                continue
            proba = (
                model.predict_test_proba(
                    local_idx, test, features[local_idx], X, method=calibration
                )
                .detach()
                .cpu()
                .numpy()
            )
            sums[test] += proba
            counts[test] += 1
        print(f"Bootstrap {start + len(group)}/{n_est}", flush=True)
        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    positive = np.full(len(y), np.nan, dtype=np.float64)
    covered = counts > 0
    positive[covered] = sums[covered] / counts[covered]
    mean_proba = np.column_stack((1 - positive, positive))
    if (~covered).any():
        print(f"{(~covered).sum()} patients have no OOB predictions.")
    return mean_proba, y.copy(), counts


def load_dataset(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        raise ValueError("Use a CSV or XLSX file.")
    if df.empty or df.columns.has_duplicates:
        raise ValueError("Data must be nonempty with unique column names.")
    aliases = {"target", "cancer status", "cancer_status", "status", "label"}
    candidates = [c for c in df if str(c).strip().lower() in aliases]
    target = next((c for c in candidates if str(c).strip().lower() == "target"), None)
    if target is None:
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous target columns: {candidates}")
        target = candidates[0] if candidates else df.columns[-1]
    negative = {
        "false",
        "0",
        "0.0",
        "healthy",
        "normal",
        "control",
        "negative",
        "no",
        "benign",
        "non-cancer",
        "noncancer",
    }
    positive = {
        "true",
        "1",
        "1.0",
        "cancer",
        "tumor",
        "tumour",
        "positive",
        "yes",
        "malignant",
        "disease",
        "colon",
        "prostate",
        "breast",
        "lung",
        "ovarian",
        "pancreatic",
        "liver",
        "kidney",
        "bladder",
        "colorectal",
        "melanoma",
        "leukemia",
        "lymphoma",
    }
    if df[target].isna().any():
        raise ValueError("Missing target labels; no samples were dropped.")
    labels = df[target].astype(str).str.strip().str.lower()
    unknown = set(labels.unique()) - negative - positive
    if unknown:
        raise ValueError(f"Unrecognized target labels: {sorted(unknown)}")
    y = labels.isin(positive).to_numpy(dtype=np.int64)
    if set(y) != {0, 1} or min(np.bincount(y)) < 4:
        raise ValueError("At least four samples of each class are required.")
    ids = {"bv name", "patient", "patient_id", "sample", "sample_id", "sample id", "id"}
    id_columns = [
        c
        for c in df
        if c != target and (str(c).strip().lower() in ids or str(c).lower().startswith("unnamed:"))
    ]
    features = df.drop(columns=[target, *id_columns])
    if features.shape[1] == 0:
        raise ValueError("No feature columns found.")
    try:
        X = features.to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be numeric. Remove non-feature metadata columns.") from exc
    if not np.isfinite(X).all():
        raise ValueError("Features contain missing, infinite, or float32-overflow values.")
    sample_ids = df[id_columns[0]].to_numpy() if id_columns else np.arange(len(df))
    print(
        f"Loaded {len(y)} samples, {X.shape[1]} features; target={target}; ID columns={id_columns}",
        flush=True,
    )
    return X, y, sample_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--networks", type=int, default=1000)
    args = parser.parse_args()
    if args.networks < 1:
        parser.error("--networks must be positive")
    X, y, sample_ids = load_dataset(args.dataset)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Bound concurrent models by a conservative memory estimate. torch.func.vmap
    # trains the independent models together on GPU (or with threaded CPU ops).
    if device == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        budget = int(free_bytes * 0.35)
        cap = 128
    else:
        budget, cap = 256 * 1024**2, 16
    per_model_bytes = 4 * (X.shape[1] * (32 * 8 + len(y) * 3 + 16 * 8) + 4096)
    model_batch = max(1, min(args.networks, cap, budget // per_model_bytes))
    print(f"Device: {device}; concurrent networks: {model_batch}", flush=True)
    probabilities, labels, counts = metwork_bootstrap(
        X,
        y,
        n_est=args.networks,
        model_batch=model_batch,
        device=device,
        random_state=23,
    )
    positive = probabilities[:, 1]
    covered = counts > 0
    sensitivities = {s: np.nan for s in [92, 94, 95, 96, 98, 100]}
    auc = np.nan
    if len(np.unique(y[covered])) == 2:
        auc = roc_auc_score(y[covered], positive[covered])
        fpr, tpr, _ = roc_curve(y[covered], positive[covered])
        sensitivities = {s: float(tpr[fpr <= 1 - s / 100 + 1e-12].max()) for s in sensitivities}
    else:
        print(
            "OOB coverage lacks both classes; AUC and sensitivities are unavailable. Increase networks."
        )
    result = {
        "Classification Algorithm": "MightyNet-bootstrap-OOB",
        "Dataset": Path(args.dataset).name,
        "# of Networks": args.networks,
        "# of Features": X.shape[1],
        "AUC": auc,
        **{f"Sensitivity at {s}% Specificity": value for s, value in sensitivities.items()},
        "Average of S@94, S@95, S@96": np.mean([sensitivities[s] for s in [94, 95, 96]]),
    }
    pd.DataFrame([result]).round(6).to_csv("Algorithm_Summary_Metrics.csv", index=False)
    print("Saved Algorithm_Summary_Metrics.csv", flush=True)


if __name__ == "__main__":
    main()
