#!/bin/bash
#SBATCH -p barbun
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_b6_res_50_80
#SBATCH -o exp_e_b6-%j.out
#SBATCH --error=exp_e_b6-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --time=3-00:00:00

# =============================================================
# Experiment E Barbun — B6: ResNet50, threads=50,60,70,80
# WARNING: ALL thread counts exceed the 20-core Slurm allocation.
#          Severe oversubscription (2.5x-4x). Performance will
#          plateau or degrade vs T20. This is NOT strong scaling.
# Est. runtime: ~4h each x 4 = ~16h
# =============================================================

echo "============================================"
echo "EXPERIMENT E Barbun B6 — ResNet50 T50,T60,T70,T80"
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
echo "Experiment E Barbun B6 finished at: $(date)"
echo "============================================"
