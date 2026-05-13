#!/bin/bash
#SBATCH -p barbun
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_b4_vgg_50_80
#SBATCH -o exp_e_b4-%j.out
#SBATCH --error=exp_e_b4-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --time=3-00:00:00

# =============================================================
# Experiment E Barbun — B4: VGG19, threads=50,60,70,80
# WARNING: ALL thread counts (50,60,70,80) EXCEED the 20-core
#          Slurm allocation. These jobs run in severe
#          OVERSUBSCRIPTION mode (2.5x-4x oversubscription).
#          Expect performance to plateau or degrade vs T20.
#          This is an oversubscription study, NOT strong scaling.
# Est. runtime: ~8h each x 4 = ~32h (plateau expected)
# =============================================================

echo "============================================"
echo "EXPERIMENT E Barbun B4 — VGG19 T50,T60,T70,T80"
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
export EXP_E_THREADS="50,60,70,80"

echo ""
echo "Environment verification:"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, Threads: {torch.get_num_threads()}')"
lscpu | grep -E "Socket|Core\(s\) per socket|Thread|Model name"
echo "Models  : $EXP_E_MODELS"
echo "Threads : $EXP_E_THREADS"
echo "WARNING: All counts exceed -c 20 allocation (oversubscription x2.5-x4)"
echo ""

cd $SLURM_SUBMIT_DIR
python train_cpu_parallel.py

echo ""
echo "============================================"
echo "Experiment E Barbun B4 finished at: $(date)"
echo "============================================"
