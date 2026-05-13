#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_vgg19_t1
#SBATCH -o exp_e_vgg19_t1-%j.out
#SBATCH --error=exp_e_vgg19_t1-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=3-00:00:00

# =============================================================
# Experiment E — VGG19, T1 only (single-thread baseline)
#
# Hardware: 2x Intel Xeon Platinum 8480+ = 112 cores/node
# Requesting minimum 56 cores (only 1 used for training).
#
# VGG19 T1 baseline is essential for computing speedup/efficiency
# across all other thread counts.
#
# Estimated runtime: ~15-30h (single thread, 15 epochs, early stop)
# Full 72h window allocated to be safe.
# =============================================================

echo "============================================"
echo "EXPERIMENT E — VGG19 T1 (baseline)"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Start     : $(date)"
echo "============================================"

module purge
module load apps/truba-ai/gpu-2024.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""

export EXP_E_MODELS="vgg19"
export EXP_E_THREADS="1"

echo ""
echo "Environment verification:"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
lscpu | grep -E "Socket|Core\(s\) per socket|Thread|Model name|NUMA"
echo "Models  : $EXP_E_MODELS"
echo "Threads : $EXP_E_THREADS"
echo ""

cd $SLURM_SUBMIT_DIR
python train_cpu_parallel.py

echo ""
echo "============================================"
echo "Experiment E VGG19 T1 finished at: $(date)"
echo "============================================"
