#!/bin/bash
#SBATCH -p barbun
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_b1_vgg_t1
#SBATCH -o exp_e_b1-%j.out
#SBATCH --error=exp_e_b1-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --time=3-00:00:00

# =============================================================
# Experiment E Barbun — B1: VGG19, threads=1 (baseline)
# Est. runtime: ~40h (based on hamsi T1 = 39h; barbun may differ)
# =============================================================

echo "============================================"
echo "EXPERIMENT E Barbun B1 — VGG19 T1"
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
export EXP_E_THREADS="1"

echo ""
echo "Environment verification:"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, Threads: {torch.get_num_threads()}')"
lscpu | grep -E "Socket|Core\(s\) per socket|Thread|Model name"
echo "Models  : $EXP_E_MODELS"
echo "Threads : $EXP_E_THREADS"
echo ""

cd $SLURM_SUBMIT_DIR
python train_cpu_parallel.py

echo ""
echo "============================================"
echo "Experiment E Barbun B1 finished at: $(date)"
echo "============================================"
