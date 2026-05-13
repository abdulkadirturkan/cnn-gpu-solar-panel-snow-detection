#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_orfoz_p3_resnet50
#SBATCH -o exp_e_orfoz_p3-%j.out
#SBATCH --error=exp_e_orfoz_p3-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112
#SBATCH --time=2-00:00:00

# =============================================================
# Experiment E — ORFOZ Part 3/3: ResNet50, all thread counts
#
# Hardware: 2x Intel Xeon Platinum 8480+ = 112 cores/node
# Thread counts: [1,2,4,7,14,28,56,112] — full socket topology
#
# ResNet50 ~50% of VGG19 cost. Includes T1 baseline so
# speedup is computed correctly within this job.
#
# Estimated runtime: ~14h+7h+4h+2.5h+2h+1.5h+0.8h+0.5h = ~32h
#                  (safe < 48h)
# =============================================================

echo "============================================"
echo "EXPERIMENT E Orfoz P3/3 — ResNet50 ALL THREADS"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Start     : $(date)"
echo "============================================"

module purge
module load apps/truba-ai/gpu-2024.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""

export EXP_E_MODELS="resnet50"
export EXP_E_THREADS="1,2,4,7,14,28,56,112"

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
echo "Experiment E Orfoz P3 finished at: $(date)"
echo "============================================"
