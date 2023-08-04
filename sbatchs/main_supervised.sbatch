#!/usr/bin/env bash
#SBATCH --mem  128G
#SBATCH --gres gpu:2
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 6
#SBATCH --constrain "khazadum"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=paulbp@kth.se
#SBATCH --output=/Midgard/home/paulbp/xview2-replication/logs/slurm/%j.out


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
# modify this to point to activate your environment
conda activate xenv
cd /Midgard/home/paulbp/xview2-replication/

srun python /Midgard/home/paulbp/xview2-replication/main.py --config-name supervised_config trainer.devices=2 data.num_workers=6 group="test_requirements"