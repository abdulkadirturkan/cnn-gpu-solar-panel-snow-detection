#!/bin/bash
#SBATCH -p barbun
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_b5_res_10_40
#SBATCH -o exp_e_b5-%j.out
#SBATCH --error=exp_e_b5-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --time=3-00:00:00

# =============================================================
# Experiment E Barbun — B5: ResNet50, threads=10,20,30,40
# WARNING: T30 and T40 exceed the 20-core Slurm allocation
#          (oversubscription). T10 and T20 are valid.
# Est. runtime: T10~7h + T20~4h + T30~4h + T40~4h = ~19h
# =============================================================

echo "============================================"
echo "EXPERIMENT E Barbun B5 — ResNet50 T10,T20,T30,T40"
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
export EXP_E_THREADS="10,20,30,40"

echo ""
echo "Environment verification:"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, Threads: {torch.get_num_threads()}')"
lscpu | grep -E "Socket|Core\(s\) per socket|Thread|Model name"
echo "Models  : $EXP_E_MODELS"
echo "Threads : $EXP_E_THREADS"
echo "NOTE: T30 and T40 exceed -c 20 allocation (oversubscription)"
echo ""

cd $SLURM_SUBMIT_DIR
python train_cpu_parallel.py

echo ""
echo "============================================"
echo "Experiment E Barbun B5 finished at: $(date)"
echo "============================================"
