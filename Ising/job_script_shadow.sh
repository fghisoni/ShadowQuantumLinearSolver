#!/bin/bash
#SBATCH --job-name=is_10
#SBATCH --partition ulow
#SBATCH -D /home/ghisoni/quantum_computing/qlsp/reviewers/Ising
#SBATCH --mem 16G
#SBATCH -t 2-23:59:59
#SBATCH --error=er_is_10
#SBATCH --output=out_is_10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francesco.ghisoni01@universitadipavia.it

# ----------- actual script to run ---------------
python SLS_potentials_v2.py
