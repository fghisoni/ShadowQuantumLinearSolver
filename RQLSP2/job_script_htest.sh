#!/bin/bash
#SBATCH --job-name=rd2ht_12
#SBATCH --partition ulow
#SBATCH -D /home/ghisoni/quantum_computing/qlsp/reviewers/RQLSP2
#SBATCH --mem 4G
#SBATCH -t 2-23:59:59
#SBATCH --error=er_rdm2ht_12
#SBATCH --output=out_rdm2ht_12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francesco.ghisoni01@universitadipavia.it

# ----------- actual script to run ---------------
python Htest_VQLS_RQLSP2.py
