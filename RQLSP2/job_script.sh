#!/bin/bash
#SBATCH --job-name=rdm2_13
#SBATCH --partition ulow
#SBATCH -D /home/ghisoni/quantum_computing/qlsp/reviewers/RQLSP2
#SBATCH --mem 4G
#SBATCH -t 2-23:59:59
#SBATCH --error=er_rdm2_13
#SBATCH --output=out_rdm2_13
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francesco.ghisoni01@universitadipavia.it

# ----------- actual script to run ---------------
python SLS_potentials_v2.py
