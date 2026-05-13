"""
Experiment E: CPU Multi-Thread Parallel Training
VGG19 and ResNet-50 for Snow Accumulation Classification on Solar Panels

This script trains both models on CPU only (no GPU) while scaling the number
of CPU threads from 1 to 56 on the TRUBA Hamsi queue. It measures strong
scaling behavior: how multi-threaded execution speeds up a fixed workload.

Threading mechanism:
  - torch.set_num_threads(N)     : PyTorch intra-op parallelism (ATen)
  - OMP_NUM_THREADS              : OpenMP backend parallelism
  - MKL_NUM_THREADS              : Intel MKL BLAS/LAPACK parallelism
  - OPENBLAS_NUM_THREADS         : OpenBLAS parallelism
  - DataLoader num_workers       : Separate processes for data loading

Platform: TRUBA Hamsi (Intel Xeon Gold 6258R, 2x28 cores = 56 per node)
"""

import os
import sys
import json
import time
import csv
import platform
import resource
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score
)
import numpy as np

# ============================================================
# 1. CONFIGURATION
# ============================================================
# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Thread counts matching Hamsi architecture: 2 sockets x 28 cores
# Can be overridden via EXP_E_THREADS env var (comma-separated), e.g. "1,2,4,7,14"
_default_threads = [1, 2, 4, 7, 14, 28, 56]
_env_threads = os.environ.get("EXP_E_THREADS", "")
THREAD_COUNTS = [int(x) for x in _env_threads.split(",") if x.strip()] if _env_threads else _default_threads

# Models to run — can be overridden via EXP_E_MODELS env var (comma-separated), e.g. "vgg19"
_default_models = ["vgg19", "resnet50"]
_env_models = os.environ.get("EXP_E_MODELS", "")
MODEL_NAMES = [m.strip() for m in _env_models.split(",") if m.strip()] if _env_models else _default_models

CONFIG = {
    "num_classes": 3,
    "class_names": ["all_snow", "no_snow", "partial"],
    "image_size": 192,
    "batch_size": 32,
    "num_epochs": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 42,
}


# ============================================================
# 2. LOGGING SETUP
# ============================================================
class Logger:
    """Dual output: stdout + log file."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def log_system_info():
    """Log CPU system specifications."""
    print("=" * 70)
    print("SYSTEM SPECIFICATIONS")
    print("=" * 70)
    print(f"Hostname        : {platform.node()}")
    print(f"Platform        : {platform.platform()}")
    print(f"CPU             : {platform.processor()}")
    cpu_count = os.cpu_count()
    print(f"CPU cores       : {cpu_count}")

    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_kb = int(line.split()[1])
                    print(f"RAM             : {mem_kb / 1024 / 1024:.1f} GB")
                    break
    except FileNotFoundError:
        print("RAM             : (could not read /proc/meminfo)")

    # CPU model from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    print(f"CPU model       : {line.split(':')[1].strip()}")
                    break
    except FileNotFoundError:
        pass

    # NUMA topology
    try:
        import subprocess
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if any(k in line.lower() for k in ["socket", "numa", "cache"]):
                    print(f"  {line.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"MKL available   : {torch.backends.mkl.is_available()}")
    print(f"OpenMP available: {torch.backends.openmp.is_available()}")
    print("=" * 70)


# ============================================================
# 3. DATA LOADING
# ============================================================
class TransformedSubset(torch.utils.data.Dataset):
    """Applies a specific transform to a subset of an ImageFolder dataset."""

    def __init__(self, dataset, indices, transform):
        self.samples = [dataset.samples[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def create_dataloaders(config, num_workers):
    """Create stratified train/val/test dataloaders."""

    transform_train = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    data_dir = config["data_dir"]
    full_dataset = datasets.ImageFolder(root=data_dir)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    targets = np.array(full_dataset.targets)
    train_idx, val_idx, test_idx = [], [], []

    for cls in range(config["num_classes"]):
        cls_indices = np.where(targets == cls)[0]
        np.random.shuffle(cls_indices)
        n = len(cls_indices)
        n_train = int(n * config["train_ratio"])
        n_val = int(n * config["val_ratio"])
        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train:n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val:])

    train_targets = targets[train_idx]
    class_counts = np.bincount(train_targets, minlength=config["num_classes"])
    class_weights = 1.0 / class_counts.astype(np.float64)
    class_weights = class_weights / class_weights.sum() * config["num_classes"]
    class_weights = torch.FloatTensor(class_weights)

    print(f"  Dataset split : Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"  Class weights : {class_weights.tolist()}")
    print(f"  Num workers   : {num_workers}")

    train_set = TransformedSubset(full_dataset, train_idx, transform_train)
    val_set = TransformedSubset(full_dataset, val_idx, transform_eval)
    test_set = TransformedSubset(full_dataset, test_idx, transform_eval)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"],
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"],
                             shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_weights, len(train_idx)


# ============================================================
# 4. MODEL DEFINITIONS (pretrained for feasible CPU training time)
# ============================================================
def build_vgg19(num_classes):
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def build_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
# 5. TRAINING & EVALUATION
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).numpy())

    return (running_loss / total, correct / total,
            np.array(all_labels), np.array(all_preds), np.array(all_probs))


def compute_dl_metrics(labels, preds, probs, class_names):
    """Set 1: DL Performance Metrics."""
    acc = np.mean(labels == preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per = f1_score(labels, preds, average=None, zero_division=0)
    prec_macro = precision_score(labels, preds, average="macro", zero_division=0)
    prec_per = precision_score(labels, preds, average=None, zero_division=0)
    rec_macro = recall_score(labels, preds, average="macro", zero_division=0)
    rec_per = recall_score(labels, preds, average=None, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_class": {class_names[i]: float(v) for i, v in enumerate(f1_per)},
        "precision_macro": float(prec_macro),
        "precision_per_class": {class_names[i]: float(v) for i, v in enumerate(prec_per)},
        "recall_macro": float(rec_macro),
        "recall_per_class": {class_names[i]: float(v) for i, v in enumerate(rec_per)},
        "auc_roc_macro": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


# ============================================================
# 6. CPU UTILIZATION MONITORING
# ============================================================
def get_cpu_times():
    """Read /proc/stat for total CPU time breakdown."""
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()  # first line: aggregate cpu stats
            parts = line.split()
            # user, nice, system, idle, iowait, irq, softirq, steal
            return {
                "user": int(parts[1]),
                "nice": int(parts[2]),
                "system": int(parts[3]),
                "idle": int(parts[4]),
                "iowait": int(parts[5]),
            }
    except (FileNotFoundError, IndexError):
        return None


def compute_cpu_utilization(before, after):
    """Compute CPU utilization percentage between two /proc/stat snapshots."""
    if not before or not after:
        return 0.0
    total_before = sum(before.values())
    total_after = sum(after.values())
    total_delta = total_after - total_before
    idle_delta = (after["idle"] + after["iowait"]) - (before["idle"] + before["iowait"])
    if total_delta == 0:
        return 0.0
    return 100.0 * (1.0 - idle_delta / total_delta)


def get_peak_memory_mb():
    """Get peak RSS memory usage in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Linux reports in KB
    except Exception:
        return 0.0


# ============================================================
# 7. SINGLE THREAD-COUNT RUN
# ============================================================
def run_single_config(model_name, build_fn, thread_count, config,
                      output_base, all_hpc_rows):
    """Train a model with a specific thread count configuration."""

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name} | THREADS: {thread_count}")
    print(f"{'='*70}")

    # --- Set threading parameters ---
    torch.set_num_threads(thread_count)
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_count)

    # Always use num_workers=0 on HPC: avoids "can only test a child process"
    # multiprocessing cleanup errors, and keeps DataLoader consistent across
    # all thread counts so OMP/MKL threads remain the only variable.
    num_workers = 0

    print(f"  torch.num_threads   : {torch.get_num_threads()}")
    print(f"  OMP_NUM_THREADS     : {os.environ['OMP_NUM_THREADS']}")
    print(f"  DataLoader workers  : {num_workers}")

    # --- Data ---
    train_loader, val_loader, test_loader, class_weights, n_train = create_dataloaders(
        config, num_workers
    )

    # --- Model ---
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    model = build_fn(config["num_classes"])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"],
                           weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=0.5, patience=3)

    # --- Warm-up pass (excluded from timing) ---
    warmup_input = torch.randn(1, 3, config["image_size"], config["image_size"])
    _ = model(warmup_input)

    # --- Training loop ---
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_times": []}
    best_val_loss = float("inf")
    early_stop_patience = 7
    epochs_no_improve = 0

    cpu_before = get_cpu_times()
    total_start = time.perf_counter()
    actual_epochs = config["num_epochs"]

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)

        epoch_time = time.perf_counter() - epoch_start
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)

        throughput = n_train / epoch_time
        print(f"  Epoch {epoch:3d}/{config['num_epochs']} | "
              f"TrLoss: {train_loss:.4f} TrAcc: {train_acc:.4f} | "
              f"VaLoss: {val_loss:.4f} VaAcc: {val_acc:.4f} | "
              f"{epoch_time:.2f}s ({throughput:.1f} img/s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no val_loss improvement for {early_stop_patience} epochs)")
                actual_epochs = epoch
                break

    total_time = time.perf_counter() - total_start
    cpu_after = get_cpu_times()
    cpu_util = compute_cpu_utilization(cpu_before, cpu_after)
    peak_mem = get_peak_memory_mb()

    # --- Test evaluation ---
    model.load_state_dict(best_state)
    test_loss, test_acc, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, criterion
    )

    # --- DL Metrics ---
    dl_metrics = compute_dl_metrics(test_labels, test_preds, test_probs,
                                    config["class_names"])
    dl_metrics["test_loss"] = float(test_loss)

    print(f"\n  TEST: Acc={dl_metrics['accuracy']:.4f} F1={dl_metrics['f1_macro']:.4f} "
          f"Prec={dl_metrics['precision_macro']:.4f} Rec={dl_metrics['recall_macro']:.4f} "
          f"AUC={dl_metrics['auc_roc_macro']:.4f}")
    print(f"  Confusion Matrix: {dl_metrics['confusion_matrix']}")

    # --- HPC Metrics ---
    avg_throughput = n_train * actual_epochs / total_time

    hpc_metrics = {
        "thread_count": thread_count,
        "num_workers": num_workers,
        "total_training_time_sec": total_time,
        "avg_epoch_time_sec": float(np.mean(history["epoch_times"])),
        "min_epoch_time_sec": float(np.min(history["epoch_times"])),
        "max_epoch_time_sec": float(np.max(history["epoch_times"])),
        "epoch_times_sec": [float(t) for t in history["epoch_times"]],
        "throughput_img_per_sec": float(avg_throughput),
        "speedup": None,
        "efficiency": None,
        "serial_fraction_estimate": None,
        "cpu_utilization_pct": float(cpu_util),
        "peak_memory_rss_mb": float(peak_mem),
    }

    print(f"  HPC: Time={total_time:.2f}s Throughput={avg_throughput:.2f} img/s "
          f"CPU_Util={cpu_util:.1f}%")

    # --- Save per-run results ---
    run_dir = os.path.join(output_base, model_name, f"threads_{thread_count}")
    set1_dir = os.path.join(run_dir, "Set1_DL_Metrics")
    set2_dir = os.path.join(run_dir, "Set2_HPC_Metrics")
    os.makedirs(set1_dir, exist_ok=True)
    os.makedirs(set2_dir, exist_ok=True)

    with open(os.path.join(set1_dir, "dl_metrics.json"), "w") as f:
        safe_dl = {k: v for k, v in dl_metrics.items() if k != "classification_report"}
        json.dump(safe_dl, f, indent=2)

    with open(os.path.join(set1_dir, "classification_report.txt"), "w") as f:
        f.write(dl_metrics["classification_report"])

    with open(os.path.join(set1_dir, "training_history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec"])
        for i in range(actual_epochs):
            writer.writerow([i + 1, history["train_loss"][i], history["train_acc"][i],
                             history["val_loss"][i], history["val_acc"][i],
                             history["epoch_times"][i]])

    with open(os.path.join(set1_dir, "confusion_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + config["class_names"])
        cm = dl_metrics["confusion_matrix"]
        for i, row in enumerate(cm):
            writer.writerow([config["class_names"][i]] + row)

    # Confusion matrix heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm_arr = np.array(dl_metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        class_names = config["class_names"]
        ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
               xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name.upper()} (threads={thread_count}) — Confusion Matrix")
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                        color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(set1_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    # Per-run loss/accuracy curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_range = range(1, actual_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(epochs_range, history["train_loss"], "b-o", markersize=3, label="Train Loss")
        ax1.plot(epochs_range, history["val_loss"], "r-o", markersize=3, label="Val Loss")
        ax1.set_yscale("log")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (log scale)")
        ax1.set_title(f"{model_name.upper()} — Loss vs Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs_range, history["train_acc"], "b-o", markersize=3, label="Train Acc")
        ax2.plot(epochs_range, history["val_acc"], "r-o", markersize=3, label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"{model_name.upper()} — Accuracy vs Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(min(min(history["train_acc"]), min(history["val_acc"])) - 0.02, 1.01)
        fig.suptitle(f"Threads={thread_count}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(set1_dir, "loss_accuracy_curves.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    with open(os.path.join(set2_dir, "hpc_metrics.json"), "w") as f:
        json.dump(hpc_metrics, f, indent=2)

    with open(os.path.join(set2_dir, "epoch_times.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "time_sec", "throughput_img_per_sec"])
        for i, t in enumerate(history["epoch_times"]):
            writer.writerow([i + 1, t, n_train / t])

    all_hpc_rows.append({
        "model": model_name,
        "thread_count": thread_count,
        "num_workers": num_workers,
        "total_time_sec": total_time,
        "avg_epoch_sec": hpc_metrics["avg_epoch_time_sec"],
        "throughput_img_s": avg_throughput,
        "accuracy": dl_metrics["accuracy"],
        "f1_macro": dl_metrics["f1_macro"],
        "precision_macro": dl_metrics["precision_macro"],
        "recall_macro": dl_metrics["recall_macro"],
        "auc_roc": dl_metrics["auc_roc_macro"],
        "cpu_util_pct": cpu_util,
        "peak_memory_mb": peak_mem,
    })

    del model, best_state
    return hpc_metrics


# ============================================================
# 8. PLOTTING
# ============================================================
def generate_plots(output_base, all_hpc_rows, config):
    """Generate summary plots for CPU thread scaling."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots.")
        return

    plots_dir = os.path.join(output_base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for model_name in ["vgg19", "resnet50"]:
        rows = [r for r in all_hpc_rows if r["model"] == model_name]
        if not rows:
            continue

        threads = [r["thread_count"] for r in rows]
        times = [r["total_time_sec"] for r in rows]
        throughputs = [r["throughput_img_s"] for r in rows]
        speedups = [r.get("speedup", 1.0) for r in rows]
        efficiencies = [r.get("efficiency", 1.0) for r in rows]
        cpu_utils = [r["cpu_util_pct"] for r in rows]
        accuracies = [r["accuracy"] for r in rows]
        f1s = [r["f1_macro"] for r in rows]
        serial_fracs = [r.get("serial_fraction", 0) for r in rows]

        max_t = max(threads)

        # --- Speedup with Amdahl curves ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(threads, speedups, "bo-", linewidth=2, markersize=8, label="Actual Speedup")
        # Ideal linear
        ax.plot(threads, threads, "r--", alpha=0.4, label="Ideal (Linear)")
        # Amdahl curves for various serial fractions
        t_range = np.linspace(1, max_t, 200)
        for fs, color in [(0.05, "green"), (0.10, "orange"), (0.20, "purple")]:
            amdahl = 1.0 / (fs + (1.0 - fs) / t_range)
            ax.plot(t_range, amdahl, "--", color=color, alpha=0.5,
                    label=f"Amdahl (fs={fs:.0%})")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Speedup (T1 / Tp)")
        ax.set_title(f"{model_name.upper()} — Speedup vs CPU Thread Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{model_name}_speedup.png"), dpi=150)
        plt.close(fig)

        # --- Efficiency plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(threads, efficiencies, "go-", linewidth=2, markersize=8)
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Ideal (100%)")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Efficiency (Speedup / Threads)")
        ax.set_title(f"{model_name.upper()} — Parallel Efficiency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.15)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{model_name}_efficiency.png"), dpi=150)
        plt.close(fig)

        # --- Throughput plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(threads, throughputs, "mo-", linewidth=2, markersize=8)
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Throughput (images/sec)")
        ax.set_title(f"{model_name.upper()} — Training Throughput vs Thread Count")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{model_name}_throughput.png"), dpi=150)
        plt.close(fig)

        # --- Training time bar chart ---
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(threads)), times,
                      tick_label=[str(t) for t in threads], color="steelblue")
        for i, t in enumerate(times):
            ax.text(i, t + max(times) * 0.01, f"{t:.1f}s",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Total Training Time (sec)")
        ax.set_title(f"{model_name.upper()} — Training Time vs Thread Count")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{model_name}_training_time.png"), dpi=150)
        plt.close(fig)

        # --- CPU Utilization ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(threads)), cpu_utils,
               tick_label=[str(t) for t in threads], color="orange")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("CPU Utilization (%)")
        ax.set_title(f"{model_name.upper()} — CPU Utilization vs Thread Count")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{model_name}_cpu_utilization.png"), dpi=150)
        plt.close(fig)

        # --- DL Metrics stability ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(threads, accuracies, "bo-", label="Accuracy", markersize=6)
        ax.plot(threads, f1s, "gs-", label="F1 (macro)", markersize=6)
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Score")
        ax.set_title(f"{model_name.upper()} — DL Metrics Stability Across Thread Counts")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{model_name}_dl_metrics_vs_threads.png"), dpi=150)
        plt.close(fig)

        # --- Loss vs Epoch (overlay all thread configs) ---
        import csv as csv_mod
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(threads)))

        for idx, tc in enumerate(threads):
            hist_path = os.path.join(output_base, model_name, f"threads_{tc}",
                                     "Set1_DL_Metrics", "training_history.csv")
            if not os.path.exists(hist_path):
                continue
            epochs_list, tr_loss, va_loss = [], [], []
            with open(hist_path) as hf:
                reader = csv_mod.DictReader(hf)
                for row in reader:
                    epochs_list.append(int(row["epoch"]))
                    tr_loss.append(float(row["train_loss"]))
                    va_loss.append(float(row["val_loss"]))
            ax1.plot(epochs_list, tr_loss, color=colors[idx], label=f"{tc} threads")
            ax2.plot(epochs_list, va_loss, color=colors[idx], label=f"{tc} threads")

        ax1.set_yscale("log")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss (log scale)")
        ax1.set_title(f"{model_name.upper()} — Train Loss")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_yscale("log")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Loss (log scale)")
        ax2.set_title(f"{model_name.upper()} — Val Loss")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{model_name}_loss_vs_epoch.png"), dpi=150)
        plt.close(fig)

        # --- Summary confusion matrix heatmap (from thread_count=1) ---
        cm_json_path = os.path.join(output_base, model_name, f"threads_{threads[0]}",
                                    "Set1_DL_Metrics", "dl_metrics.json")
        if os.path.exists(cm_json_path):
            with open(cm_json_path) as cmf:
                dl_data = json.load(cmf)
            cm_arr = np.array(dl_data["confusion_matrix"])
            class_names = config["class_names"]
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
                   xticklabels=class_names, yticklabels=class_names)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{model_name.upper()} — Confusion Matrix (Test Set)")
            for i in range(cm_arr.shape[0]):
                for j in range(cm_arr.shape[1]):
                    ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                            color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
                            fontsize=14, fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"{model_name}_confusion_matrix.png"), dpi=150)
            plt.close(fig)

        # --- Epoch time boxplot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        epoch_data = []
        for tc in threads:
            hist_path = os.path.join(output_base, model_name, f"threads_{tc}",
                                     "Set1_DL_Metrics", "training_history.csv")
            if not os.path.exists(hist_path):
                epoch_data.append([])
                continue
            ep_times = []
            with open(hist_path) as hf:
                reader = csv_mod.DictReader(hf)
                for row in reader:
                    ep_times.append(float(row["epoch_time_sec"]))
            epoch_data.append(ep_times)

        if any(epoch_data):
            ax.boxplot(epoch_data, labels=[str(t) for t in threads])
            ax.set_xlabel("Thread Count")
            ax.set_ylabel("Epoch Time (sec)")
            ax.set_title(f"{model_name.upper()} — Epoch Time Distribution")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"{model_name}_epoch_time_boxplot.png"), dpi=150)
        plt.close(fig)

    # --- Combined model comparison ---
    vgg_rows = [r for r in all_hpc_rows if r["model"] == "vgg19"]
    res_rows = [r for r in all_hpc_rows if r["model"] == "resnet50"]
    if vgg_rows and res_rows:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        vgg_t = [r["thread_count"] for r in vgg_rows]
        res_t = [r["thread_count"] for r in res_rows]

        ax = axes[0]
        ax.plot(vgg_t, [r.get("speedup", 1) for r in vgg_rows], "bo-", label="VGG19")
        ax.plot(res_t, [r.get("speedup", 1) for r in res_rows], "rs-", label="ResNet50")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Speedup")
        ax.set_title("Speedup Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(vgg_t, [r["throughput_img_s"] for r in vgg_rows], "bo-", label="VGG19")
        ax.plot(res_t, [r["throughput_img_s"] for r in res_rows], "rs-", label="ResNet50")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Throughput (img/s)")
        ax.set_title("Throughput Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(vgg_t, [r.get("efficiency", 1) for r in vgg_rows], "bo-", label="VGG19")
        ax.plot(res_t, [r.get("efficiency", 1) for r in res_rows], "rs-", label="ResNet50")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Efficiency")
        ax.set_title("Efficiency Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle("VGG19 vs ResNet50 — CPU Thread Scaling Comparison",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "combined_comparison.png"), dpi=150)
        plt.close(fig)

    print(f"\nPlots saved to: {plots_dir}")


# ============================================================
# 9. MAIN
# ============================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(script_dir, f"ExpE_run_{timestamp}")
    os.makedirs(output_base, exist_ok=True)

    CONFIG["data_dir"] = os.path.join(script_dir, "..", "Data")

    log_path = os.path.join(script_dir, f"experiment_e_{timestamp}.log")
    logger = Logger(log_path)
    sys.stdout = logger

    print(f"Experiment E started at: {datetime.now().isoformat()}")
    print(f"Output directory: {output_base}")
    print(f"Log file: {log_path}")
    print(f"\nDevice: CPU only (CUDA_VISIBLE_DEVICES disabled)")
    log_system_info()

    print("\nHyperparameters:")
    for k, v in CONFIG.items():
        if k not in ("data_dir",):
            print(f"  {k}: {v}")
    print(f"  thread_counts: {THREAD_COUNTS}")

    # --- Run all configurations ---
    all_hpc_rows = []
    _all_model_configs = [
        ("vgg19", build_vgg19),
        ("resnet50", build_resnet50),
    ]
    model_configs = [(n, fn) for n, fn in _all_model_configs if n in MODEL_NAMES]
    print(f"  models: {[n for n, _ in model_configs]}")

    for model_name, build_fn in model_configs:
        for thread_count in THREAD_COUNTS:
            run_single_config(model_name, build_fn, thread_count, CONFIG,
                              output_base, all_hpc_rows)

    # --- Compute speedup, efficiency, serial fraction ---
    for model_name in ["vgg19", "resnet50"]:
        rows = [r for r in all_hpc_rows if r["model"] == model_name]
        baseline_row = [r for r in rows if r["thread_count"] == 1]
        if baseline_row:
            t1 = baseline_row[0]["total_time_sec"]
            for r in rows:
                p = r["thread_count"]
                r["speedup"] = t1 / r["total_time_sec"]
                r["efficiency"] = r["speedup"] / p
                # Estimate serial fraction from Amdahl's law
                if p > 1:
                    s = r["speedup"]
                    r["serial_fraction"] = (1.0 / s - 1.0 / p) / (1.0 - 1.0 / p)
                else:
                    r["serial_fraction"] = 0.0

    # --- Save combined summary CSV ---
    summary_csv = os.path.join(output_base, "experiment_e_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        if all_hpc_rows:
            writer = csv.DictWriter(f, fieldnames=all_hpc_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_hpc_rows)
    print(f"\nSummary CSV saved to: {summary_csv}")

    summary_json = os.path.join(output_base, "experiment_e_summary.json")
    with open(summary_json, "w") as f:
        json.dump(all_hpc_rows, f, indent=2)

    # --- Generate plots ---
    generate_plots(output_base, all_hpc_rows, CONFIG)

    # --- Print final summary ---
    print(f"\n{'='*70}")
    print("EXPERIMENT E FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'Threads':>8} {'Time(s)':>10} {'Speedup':>8} {'Effic':>8} "
          f"{'Thruput':>10} {'Acc':>7} {'F1':>7} {'fs':>8}")
    print("-" * 90)
    for r in all_hpc_rows:
        print(f"{r['model']:<10} {r['thread_count']:>8} {r['total_time_sec']:>10.2f} "
              f"{r.get('speedup', 0):>8.3f} {r.get('efficiency', 0):>8.4f} "
              f"{r['throughput_img_s']:>10.2f} {r['accuracy']:>7.4f} "
              f"{r['f1_macro']:>7.4f} {r.get('serial_fraction', 0):>8.4f}")

    print(f"\nExperiment E completed at: {datetime.now().isoformat()}")
    print(f"All results in: {output_base}")

    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
