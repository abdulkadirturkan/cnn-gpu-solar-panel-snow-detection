
# CNN GPU-Accelerated Solar Panel Snow Detection

This repository contains the implementation and experimental framework of a GPU-accelerated deep learning system for **snow detection on solar panels** using CNN architectures (VGG-19 and ResNet-50). The project focuses on **CPU multi-threading vs GPU acceleration performance analysis** on HPC infrastructure (TRUBA).

---

## Objective

- CPU multi-threaded vs GPU training performance analysis
- Amdahl's Law empirical validation on real HPC systems
- Architectural impact of VGG-19 vs ResNet-50 on parallel scaling
- Snow detection as the application domain

---

## Models

- VGG-19 (ImageNet pretrained)
- ResNet-50 (ImageNet pretrained)

Input:

- 192×192 RGB imagesOutput:
- 3 classes:
  - `all_snow`
  - `no_snow`
  - `partial`

---

## Experimental Setup

### Hardware

- Intel Xeon Platinum 8480+ (112 threads)
- NVIDIA V100 SXM2 16GB (5120 CUDA cores)
- TRUBA HPC cluster

### Software

- Python 3.12.2
- PyTorch 2.1+
- CUDA 12.0
- OpenMP / MKL

---

## Experiments

### CPU Experiments

- Single-thread baseline
- Multi-thread scaling: {1, 2, 4, 7, 14, 28, 56, 112} threads

### GPU Experiments

- V100 acceleration tests
- Batch-based training comparison

---

## Key Results

- VGG-19 CPU speedup: **up to 28×**
- ResNet-50 CPU speedup: **~1.56×**
- GPU speedup vs 1-thread CPU:
  - ResNet-50: **163×**
  - VGG-19: **1067×**
- GPU maintains consistent performance across architectures

---

## Repository Structure

```text
.
├── src/ # Training scripts
├── slurm/ # HPC job scripts
 ├── cpu/
 └── gpu/
├── results/ # Aggregated results (CSV, JSON)
 └── plots/ # Figures for paper/thesis
├── logs/ # SLURM output logs
├── docs/ # Methodology & math
├── README.md             # Read Me
└── README_TR.md          # Read Me in Turkish
```

---

## Methodology

- Data augmentation (flip, rotation, jitter)
- Weighted Cross Entropy loss
- Adam optimizer (lr=1e-4)
- ReduceLROnPlateau scheduler
- Early stopping (patience=7)

---

## Notes

- This repository does NOT contain raw dataset images
- Only processed results and logs are stored
- Designed for reproducibility in HPC environments

---

## Citation

Turkan, A., Hangun, B. and Eyecioglu, O. (2026). HPC-Accelerated CNN
Training for Solar Panel Snow Detection: A Comparative Analysis of
Multi-threaded CPU and GPU Performance with VGG-19 and ResNet-50.
International Journal of Smart Grid. (in press)

---

Türkçe dokümantasyon için: [README_TR.md](README_TR.md)
