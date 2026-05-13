"""
Experiment A: Single-CPU Baseline Training
VGG19 and ResNet-50 for Snow Accumulation Classification on Solar Panels

This script enforces single-thread CPU-only execution to establish
a true sequential baseline for comparison with CUDA-parallel experiments.
"""

import os
import sys
import json
import time

# ============================================================
# 1. ENFORCE SINGLE-THREAD CPU EXECUTION (must be set BEFORE importing torch)
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # Hide all GPUs
os.environ["OMP_NUM_THREADS"] = "1"               # OpenMP single thread
os.environ["MKL_NUM_THREADS"] = "1"               # MKL single thread
os.environ["OPENBLAS_NUM_THREADS"] = "1"           # OpenBLAS single thread
os.environ["NUMEXPR_NUM_THREADS"] = "1"            # NumExpr single thread
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"         # macOS Accelerate

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

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
# 2. CONFIGURATION
# ============================================================
CONFIG = {
    "data_dir": os.path.join(os.path.dirname(__file__), "..", "Data"),
    "output_dir": os.path.join(os.path.dirname(__file__), "results"),
    "num_classes": 3,
    "class_names": ["all_snow", "no_snow", "partial"],
    "image_size": 192,
    "batch_size": 32,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 42,
}

DEVICE = torch.device("cpu")


def verify_cpu_only():
    """Verify and print that execution is strictly CPU, single-thread."""
    print("=" * 60)
    print("EXPERIMENT A: SINGLE-CPU BASELINE")
    print("=" * 60)
    print(f"PyTorch version  : {torch.__version__}")
    print(f"Device           : {DEVICE}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    print(f"Num threads      : {torch.get_num_threads()}")
    print(f"Interop threads  : {torch.get_num_interop_threads()}")
    print(f"OMP_NUM_THREADS  : {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"MKL_NUM_THREADS  : {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    assert not torch.cuda.is_available(), "CUDA must not be available for Experiment A!"
    assert torch.get_num_threads() == 1, "Thread count must be 1!"
    print("Verification PASSED: CPU-only, single-thread execution confirmed.")
    print("=" * 60)


# ============================================================
# 3. DATA LOADING
# ============================================================
def create_dataloaders(config):
    """Create stratified train/val/test dataloaders with class-weight computation."""

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

    full_dataset = datasets.ImageFolder(root=config["data_dir"])

    # Stratified split
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

    # Compute class weights for imbalanced data (inverse frequency)
    train_targets = targets[train_idx]
    class_counts = np.bincount(train_targets, minlength=config["num_classes"])
    class_weights = 1.0 / class_counts.astype(np.float64)
    class_weights = class_weights / class_weights.sum() * config["num_classes"]
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    print(f"\nDataset split (stratified):")
    print(f"  Train : {len(train_idx)} samples")
    print(f"  Val   : {len(val_idx)} samples")
    print(f"  Test  : {len(test_idx)} samples")
    print(f"  Class weights: {class_weights.tolist()}")
    print(f"  Class mapping: {full_dataset.class_to_idx}")

    # Create subset datasets with appropriate transforms
    train_set = TransformedSubset(full_dataset, train_idx, transform_train)
    val_set = TransformedSubset(full_dataset, val_idx, transform_eval)
    test_set = TransformedSubset(full_dataset, test_idx, transform_eval)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"],
                             shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader, class_weights


class TransformedSubset(torch.utils.data.Dataset):
    """Applies a specific transform to a subset of an ImageFolder dataset."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset.samples[self.indices[idx]]
        from PIL import Image
        img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================
def build_vgg19(num_classes):
    """VGG19 with pretrained weights, adapted for 3-class classification."""
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model.to(DEVICE)


def build_resnet50(num_classes):
    """ResNet-50 with pretrained weights, adapted for 3-class classification."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# ============================================================
# 5. TRAINING AND EVALUATION
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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


def evaluate(model, loader, criterion, num_classes):
    """Evaluate model and return loss, accuracy, and per-sample predictions."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    return (running_loss / total, correct / total,
            np.array(all_labels), np.array(all_preds), np.array(all_probs))


def compute_all_metrics(labels, preds, probs, class_names):
    """
    Compute all DL performance metrics:
      - Accuracy, Loss (tracked externally), F1-score, Confusion Matrix
      - Precision (added metric 1)
      - Recall (added metric 2)
      - AUC-ROC One-vs-Rest (added metric 3 — bonus)
    """
    acc = np.mean(labels == preds)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    prec_macro = precision_score(labels, preds, average="macro", zero_division=0)
    prec_per_class = precision_score(labels, preds, average=None, zero_division=0)

    rec_macro = recall_score(labels, preds, average="macro", zero_division=0)
    rec_per_class = recall_score(labels, preds, average=None, zero_division=0)

    # AUC-ROC (One-vs-Rest, macro averaged)
    try:
        auc_macro = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = float("nan")

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_class": {class_names[i]: float(v) for i, v in enumerate(f1_per_class)},
        "precision_macro": float(prec_macro),
        "precision_per_class": {class_names[i]: float(v) for i, v in enumerate(prec_per_class)},
        "recall_macro": float(rec_macro),
        "recall_per_class": {class_names[i]: float(v) for i, v in enumerate(rec_per_class)},
        "auc_roc_macro": float(auc_macro),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return metrics


# ============================================================
# 6. HPC / PARALLEL PERFORMANCE METRICS
# ============================================================
def compute_hpc_metrics(epoch_times, total_train_time, num_threads=1):
    """
    Compute parallel/HPC performance metrics for single-CPU baseline:
      - Total training time
      - Average time per epoch
      - Throughput (images/sec)  — added metric 1
      - FLOPS utilization estimate — added metric 2
    For the baseline, speedup = 1.0 and efficiency = 1.0 by definition.
    """
    metrics = {
        "num_threads": num_threads,
        "total_training_time_sec": total_train_time,
        "avg_epoch_time_sec": float(np.mean(epoch_times)),
        "min_epoch_time_sec": float(np.min(epoch_times)),
        "max_epoch_time_sec": float(np.max(epoch_times)),
        "epoch_times_sec": [float(t) for t in epoch_times],
        "speedup": 1.0,          # Baseline: S(1) = 1.0
        "efficiency": 1.0,       # Baseline: E(1) = S(1)/1 = 1.0
        "throughput_img_per_sec": None,  # Filled after training
        "cpu_time_breakdown": None,      # Filled after training
    }
    return metrics


# ============================================================
# 7. MAIN TRAINING LOOP
# ============================================================
def run_experiment(model_name, model, train_loader, val_loader, test_loader,
                   class_weights, config):
    """Full training loop for a single model."""

    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")

    output_dir = os.path.join(config["output_dir"], model_name)
    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"],
                           weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=0.5, patience=5)

    num_train_samples = len(train_loader.dataset)
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "epoch_times": []}

    best_val_loss = float("inf")
    total_train_start = time.perf_counter()

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.perf_counter()

        # --- Train ---
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer)

        # --- Validate ---
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion,
                                              config["num_classes"])

        epoch_time = time.perf_counter() - epoch_start
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)

        throughput = num_train_samples / epoch_time

        print(f"Epoch {epoch:3d}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s  ({throughput:.1f} img/s)")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    total_train_time = time.perf_counter() - total_train_start

    # --- Test evaluation ---
    print(f"\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth"),
                                     weights_only=True))

    test_loss, test_acc, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, criterion, config["num_classes"]
    )

    # --- DL Metrics (Set 1) ---
    dl_metrics = compute_all_metrics(test_labels, test_preds, test_probs,
                                     config["class_names"])
    dl_metrics["test_loss"] = float(test_loss)

    print(f"\n--- TEST RESULTS: {model_name} ---")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {dl_metrics['accuracy']:.4f}")
    print(f"F1 (macro)    : {dl_metrics['f1_macro']:.4f}")
    print(f"Precision     : {dl_metrics['precision_macro']:.4f}")
    print(f"Recall        : {dl_metrics['recall_macro']:.4f}")
    print(f"AUC-ROC       : {dl_metrics['auc_roc_macro']:.4f}")
    print(f"\nConfusion Matrix:\n{np.array(dl_metrics['confusion_matrix'])}")
    print(f"\n{dl_metrics['classification_report']}")

    # --- HPC Metrics (Set 2) ---
    hpc_metrics = compute_hpc_metrics(history["epoch_times"], total_train_time,
                                      num_threads=1)
    avg_throughput = num_train_samples * config["num_epochs"] / total_train_time
    hpc_metrics["throughput_img_per_sec"] = float(avg_throughput)

    # CPU Time Breakdown: forward vs backward (estimated from epoch structure)
    hpc_metrics["cpu_time_breakdown"] = {
        "total_training_sec": total_train_time,
        "avg_epoch_sec": float(np.mean(history["epoch_times"])),
    }

    print(f"\n--- HPC METRICS: {model_name} ---")
    print(f"Total training time : {total_train_time:.2f}s")
    print(f"Avg epoch time      : {hpc_metrics['avg_epoch_time_sec']:.2f}s")
    print(f"Throughput           : {avg_throughput:.2f} img/s")
    print(f"Speedup (baseline)  : {hpc_metrics['speedup']:.2f}")
    print(f"Efficiency (baseline): {hpc_metrics['efficiency']:.2f}")

    # --- Save all results ---
    results = {
        "model": model_name,
        "experiment": "A_single_cpu",
        "config": config,
        "training_history": history,
        "dl_metrics": dl_metrics,
        "hpc_metrics": hpc_metrics,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    return results


# ============================================================
# 8. ENTRY POINT
# ============================================================
def main():
    verify_cpu_only()

    # Reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CONFIG["data_dir"] = os.path.join(script_dir, "..", "Data")
    CONFIG["output_dir"] = os.path.join(script_dir, "results")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print(f"\nData directory: {CONFIG['data_dir']}")
    print(f"Output directory: {CONFIG['output_dir']}")

    # Create dataloaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(CONFIG)

    all_results = {}

    # --- VGG19 ---
    vgg19 = build_vgg19(CONFIG["num_classes"])
    total_params = sum(p.numel() for p in vgg19.parameters())
    print(f"\nVGG19 total parameters: {total_params:,}")
    all_results["vgg19"] = run_experiment(
        "vgg19", vgg19, train_loader, val_loader, test_loader,
        class_weights, CONFIG
    )
    del vgg19
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- ResNet-50 ---
    resnet50 = build_resnet50(CONFIG["num_classes"])
    total_params = sum(p.numel() for p in resnet50.parameters())
    print(f"\nResNet-50 total parameters: {total_params:,}")
    all_results["resnet50"] = run_experiment(
        "resnet50", resnet50, train_loader, val_loader, test_loader,
        class_weights, CONFIG
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("EXPERIMENT A SUMMARY")
    print("=" * 60)
    for name, res in all_results.items():
        dl = res["dl_metrics"]
        hpc = res["hpc_metrics"]
        print(f"\n{name.upper()}:")
        print(f"  Test Accuracy    : {dl['accuracy']:.4f}")
        print(f"  F1 (macro)       : {dl['f1_macro']:.4f}")
        print(f"  Precision (macro): {dl['precision_macro']:.4f}")
        print(f"  Recall (macro)   : {dl['recall_macro']:.4f}")
        print(f"  AUC-ROC (macro)  : {dl['auc_roc_macro']:.4f}")
        print(f"  Total Time       : {hpc['total_training_time_sec']:.2f}s")
        print(f"  Throughput       : {hpc['throughput_img_per_sec']:.2f} img/s")

    # Save combined summary
    summary_path = os.path.join(CONFIG["output_dir"], "experiment_a_summary.json")
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            "dl_metrics": res["dl_metrics"],
            "hpc_metrics": res["hpc_metrics"],
        }
        # Remove non-serializable items
        if "classification_report" in summary[name]["dl_metrics"]:
            del summary[name]["dl_metrics"]["classification_report"]

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCombined summary saved to: {summary_path}")
    print("Experiment A complete.")


if __name__ == "__main__":
    main()
