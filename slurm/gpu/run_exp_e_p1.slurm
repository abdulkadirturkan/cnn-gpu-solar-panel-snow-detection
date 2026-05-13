#!/bin/bash
#SBATCH -p hamsi
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_p1_vgg_slow
#SBATCH -o exp_e_p1-%j.out
#SBATCH --error=exp_e_p1-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=3-00:00:00

# =============================================================
# Experiment E — Part 1/3: VGG19, threads 1 and 2
# Estimated runtime: ~61h (T1~39h + T2~22h with num_workers=0)
# =============================================================

echo "============================================"
echo "EXPERIMENT E Part 1/3 — VGG19 T1,T2"
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
export EXP_E_THREADS="1,2"

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
echo "Experiment E Part 1 finished at: $(date)"
echo "============================================"
