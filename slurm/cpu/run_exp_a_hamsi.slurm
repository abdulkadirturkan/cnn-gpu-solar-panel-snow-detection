#!/bin/bash
#SBATCH -p hamsi                        # Hamsi CPU kuyurugu
#SBATCH -A oeyecioglu                   # Kullanici adi
#SBATCH -J exp_a_cpu_baseline           # Is ismi
#SBATCH -o exp_a_cpu-%j.out             # Cikti dosyasi
#SBATCH --error=exp_a_cpu-%j.err        # Hata dosyasi
#SBATCH -N 1                            # Tek node
#SBATCH -n 1                            # Tek gorev
#SBATCH -c 56                           # Hamsi node basina 56 cekirdek zorunlu; thread limiti Python tarafinda zorlanir
#SBATCH --time=3-00:00:00              # 3 gun (hamsi max)

# =============================================================
# Experiment A: Single-CPU Baseline
# VGG19 + ResNet-50 — strictly 1 CPU core, 0 GPUs
# =============================================================

echo "============================================"
echo "EXPERIMENT A: Single-CPU Baseline (Hamsi)"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Start     : $(date)"
echo "============================================"

# --- Modul ortami ---
module purge
module load apps/truba-ai/gpu-2024.0
export PYTHONNOUSERSITE=1

# --- Tek thread zorunlulugu ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

echo ""
echo "Environment verification:"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Threads: {torch.get_num_threads()}')"
echo ""

# --- Calistir ---
cd $SLURM_SUBMIT_DIR
python train_cpu.py

echo ""
echo "============================================"
echo "Experiment A finished at: $(date)"
echo "============================================"
