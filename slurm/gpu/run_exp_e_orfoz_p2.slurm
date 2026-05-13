#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_orfoz_p2_vgg_fast
#SBATCH -o exp_e_orfoz_p2-%j.out
#SBATCH --error=exp_e_orfoz_p2-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112
#SBATCH --time=1-12:00:00

# =============================================================
# Experiment E — ORFOZ Part 2/3: VGG19, threads=[7,14,28,56,112]
#
# Hardware: 2x Intel Xeon Platinum 8480+ = 112 cores/node
# NUMA topology: 2 sockets x 56 cores each
# Thread design:
#   T7,T14 : sub-socket scaling
#   T28     : half-socket (NUMA node 0, half)
#   T56     : 1 full socket — NUMA boundary
#   T112    : full node (both sockets)
#
# Estimated runtime: T7~5h + T14~4h + T28~2.5h + T56~1.5h + T112~1h
#                  = ~14h (safe < 36h)
# NOTE: T1 baseline is in Part 1. For speedup, combine with P1 results.
# =============================================================

echo "============================================"
echo "EXPERIMENT E Orfoz P2/3 — VGG19 T7,T14,T28,T56,T112"
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
export EXP_E_THREADS="7,14,28,56,112"

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
echo "Experiment E Orfoz P2 finished at: $(date)"
echo "============================================"
