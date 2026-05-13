#!/bin/bash
#SBATCH -p barbun
#SBATCH -A oeyecioglu
#SBATCH -J exp_e_b3_vgg_10_40
#SBATCH -o exp_e_b3-%j.out
#SBATCH --error=exp_e_b3-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --time=3-00:00:00

# =============================================================
# Experiment E Barbun — B3: VGG19, threads=10,20,30,40
# WARNING: T10 and T20 are within the 20-core allocation.
#          T30 and T40 EXCEED the 20-core Slurm allocation
#          and will run in OVERSUBSCRIPTION mode (thread
#          count > physical cores). Results for T30/T40
#          may show degraded performance. See analysis notes.
# Est. runtime: T10~14h + T20~8h + T30~8h + T40~8h = ~38h
# =============================================================

echo "============================================"
echo "EXPERIMENT E Barbun B3 — VGG19 T10,T20,T30,T40"
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
echo "Experiment E Barbun B3 finished at: $(date)"
echo "============================================"
