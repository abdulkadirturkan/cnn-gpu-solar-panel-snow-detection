#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_vgg19_t2t4t7
#SBATCH -o exp_e_vgg19_t2t4t7-%j.out
#SBATCH --error=exp_e_vgg19_t2t4t7-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=1-12:00:00

# =============================================================
# Experiment E — VGG19, T2, T4, T7
#
# Hardware: 2x Intel Xeon Platinum 8480+ = 112 cores/node
# Requesting minimum 56 cores (max 7 used for training).
#
# These sub-socket thread counts fill the gap between T1 baseline
# (separate job) and T14-T112 (already completed).
#
# Estimated runtime: T2~10h + T4~7h + T7~5h = ~22h
# 36h window allocated for safety margin.
# =============================================================

echo "============================================"
echo "EXPERIMENT E — VGG19 T2, T4, T7"
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
export EXP_E_THREADS="2,4,7"

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
echo "Experiment E VGG19 T2,T4,T7 finished at: $(date)"
echo "============================================"
