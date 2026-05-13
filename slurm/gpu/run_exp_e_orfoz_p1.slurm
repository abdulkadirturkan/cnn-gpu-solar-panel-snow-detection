#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_orfoz_p1_vgg_slow
#SBATCH -o exp_e_orfoz_p1-%j.out
#SBATCH --error=exp_e_orfoz_p1-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112
#SBATCH --time=3-00:00:00

# =============================================================
# Experiment E — ORFOZ Part 1/3: VGG19, threads=[1,2,4]
#
# Hardware: 2x Intel Xeon Platinum 8480+ = 112 cores/node
# NUMA topology: 2 sockets x 56 cores each
# Thread design: [1,2,4] sub-socket scaling
#
# Why orfoz: No oversubscription up to T112. Sapphire Rapids
# architecture, fastest CPU on TRUBA. 504 nodes available.
#
# Estimated runtime: T1~25h + T2~13h + T4~8h = ~46h (safe < 72h)
# (Xeon Platinum 8480+ ~1.5-2x faster per core vs hamsi Gold 6338)
# =============================================================

echo "============================================"
echo "EXPERIMENT E Orfoz P1/3 — VGG19 T1,T2,T4"
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
export EXP_E_THREADS="1,2,4"

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
echo "Experiment E Orfoz P1 finished at: $(date)"
echo "============================================"
