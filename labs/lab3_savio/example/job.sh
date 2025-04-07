#!/bin/bash
# Job name:
#SBATCH --job-name=saahit_lab3_run
#
# Account:
#SBATCH --account=fc_dweisz
#
# Partition:
#SBATCH --partition=savio2_1080ti
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case):
#SBATCH --ntasks=1
#
# Processors per task:
# Always at least twice the number of GPUs
#SBATCH --cpus-per-task=2
#
#Number of GPUs
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=00:10:00
#
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smogan@berkeley.edu

module load python
module load ml/pytorch
module load cuda

python cifar10_example.py

# Finally run on the cluster using: sbatch job.sh