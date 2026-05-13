#!/bin/bash
#SBATCH -p barbun                       # Barbun CPU kuyurugu
#SBATCH -A oeyecioglu                   # Kullanici adi
#SBATCH -J exp_e_cpu_low                # Is ismi
#SBATCH -o exp_e_cpu_low-%j.out         # Cikti dosyasi
#SBATCH --error=exp_e_cpu_low-%j.err    # Hata dosyasi
#SBATCH -N 1                            # Tek node
#SBATCH -n 1                            # Tek gorev
#SBATCH -c 20                           # Barbun: 20 cekirdek zorunlu
#SBATCH --time=2-00:00:00               # 2 gun

# =============================================================
# Experiment E (Part 1/2): CPU Thread Scaling — Low Thread Counts
# VGG19 + ResNet-50 — 1..14 threads on barbun (20 cores)
# =============================================================

echo "============================================"
echo "EXPERIMENT E (Part 1): Low Thread Counts"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Start     : $(date)"
echo "============================================"

module purge
module load apps/truba-ai/gpu-2024.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""

# Only run thread counts that fit within 20 barbun cores
export EXP_E_THREADS="1,2,4,7,14"

echo ""
echo "Environment verification:"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Threads: {torch.get_num_threads()}')"
echo "Thread counts: $EXP_E_THREADS"
echo ""

cd $SLURM_SUBMIT_DIR
python train_cpu_parallel.py

echo ""
echo "============================================"
echo "Experiment E (Part 1) finished at: $(date)"
echo "============================================"
