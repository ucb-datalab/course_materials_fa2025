#!/bin/bash
# Job name:
#SBATCH --job-name=saahit_lecture_example_run
#
# Account:
#SBATCH --account=ic_ay128f25
#
# Partition:
#SBATCH --partition=savio2_htc
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks:
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=1
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

python cifar10_example.py

# Finally run on the cluster using: sbatch