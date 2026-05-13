#!/bin/bash
#SBATCH -p hamsi
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_p2_vgg_fast
#SBATCH -o exp_e_p2-%j.out
#SBATCH --error=exp_e_p2-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=1-12:00:00

# =============================================================
# Experiment E — Part 2/3: VGG19, threads 4,7,14,28,56
# Estimated runtime: ~30h
# NOTE: Speedup/efficiency vs T1 must be computed in post-
# processing using the T1 time from Part 1's hpc_metrics.json.
# =============================================================

echo "============================================"
echo "EXPERIMENT E Part 2/3 — VGG19 T4..T56"
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
export EXP_E_THREADS="4,7,14,28,56"

echo ""
echo "Environment verification:"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo "Models  : $EXP_E_MODELS"
echo "Threads : $EXP_E_THREADS"
echo ""

cd $SLURM_SUBMIT_DIR
python train_cpu_parallel.py

echo ""
echo "============================================"
echo "Experiment E Part 2 finished at: $(date)"
echo "============================================"
