#!/bin/bash
#SBATCH --job-name=rdmht_10
#SBATCH --partition ulow
#SBATCH -D /home/ghisoni/quantum_computing/qlsp/reviewers/RQLSP1
#SBATCH --mem 4G
#SBATCH -t 2-23:59:59
#SBATCH --error=er_rdmht_10
#SBATCH --output=out_rdmht_10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francesco.ghisoni01@universitadipavia.it

# ----------- actual script to run ---------------
python Htest_VQLS_RQLSP.py
