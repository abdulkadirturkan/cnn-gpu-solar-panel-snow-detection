#!/bin/bash
#SBATCH -p hamsi                        # Hamsi CPU kuyurugu
#SBATCH -A oeyecioglu                   # Kullanici adi
#SBATCH -J exp_e_cpu_parallel           # Is ismi
#SBATCH -o exp_e_cpu-%j.out             # Cikti dosyasi
#SBATCH --error=exp_e_cpu-%j.err        # Hata dosyasi
#SBATCH -N 1                            # Tek node
#SBATCH -n 1                            # Tek gorev
#SBATCH -c 56                           # Hamsi node basina 56 cekirdek zorunlu
#SBATCH --time=3-00:00:00              # 3 gun — 2 model x 7 thread config x 15 epoch

# =============================================================
# Experiment E: CPU Multi-Thread Parallel Training (All Configs)
# VGG19 + ResNet-50 — Thread counts 1,2,4,7,14,28,56 on hamsi
# Single node, single job for consistent speedup baseline
# =============================================================

echo "============================================"
echo "EXPERIMENT E: CPU Thread Scaling (Full)"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Start     : $(date)"
echo "============================================"

# --- Modul ortami ---
module purge
module load apps/truba-ai/gpu-2024.0
export PYTHONNOUSERSITE=1

# --- GPU'yu devre disi birak ---
export CUDA_VISIBLE_DEVICES=""

# All thread counts on same hardware for valid speedup comparison
export EXP_E_THREADS="1,2,4,7,14,28,56"

echo ""
echo "Environment verification:"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Threads: {torch.get_num_threads()}')"
lscpu | grep -E "Socket|Core|Thread|Model name|CPU MHz"
echo "Thread counts: $EXP_E_THREADS"
echo ""

# --- Calistir ---
cd $SLURM_SUBMIT_DIR
python train_cpu_parallel.py

echo ""
echo "============================================"
echo "Experiment E finished at: $(date)"
echo "============================================"
