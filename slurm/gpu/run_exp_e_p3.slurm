#!/bin/bash
#SBATCH -p hamsi
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_p3_resnet50
#SBATCH -o exp_e_p3-%j.out
#SBATCH --error=exp_e_p3-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=2-00:00:00

# =============================================================
# Experiment E — Part 3/3: ResNet50, all thread counts
# Estimated runtime: ~46h (ResNet50 ~50% of VGG19 cost)
# Includes T1 baseline so speedup is computed internally.
# =============================================================

echo "============================================"
echo "EXPERIMENT E Part 3/3 — ResNet50 ALL THREADS"
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
export EXP_E_THREADS="1,2,4,7,14,28,56"

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
echo "Experiment E Part 3 finished at: $(date)"
echo "============================================"
