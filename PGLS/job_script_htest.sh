#!/bin/bash
#SBATCH --job-name=pght_11
#SBATCH --partition ulow
#SBATCH -D /home/ghisoni/quantum_computing/qlsp/reviewers/PGLS
#SBATCH --mem 4G
#SBATCH -t 2-23:59:59
#SBATCH --error=er_pght_11
#SBATCH --output=out_pght_11
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francesco.ghisoni01@universitadipavia.it

# ----------- actual script to run ---------------
python Htest_VQLS_PGLS.py
