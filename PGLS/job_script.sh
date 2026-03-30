#!/bin/bash
#SBATCH --job-name=pg_8qubit
#SBATCH --partition ulow
#SBATCH -D /home/ghisoni/quantum_computing/qlsp/reviewers/PGLS
#SBATCH --mem 128G
#SBATCH -t 2-23:59:59
#SBATCH --error=er_pg_8qubit
#SBATCH --output=out_pg_8qubit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francesco.ghisoni01@universitadipavia.it

# ----------- actual script to run ---------------
python SLS_potentials_v2_8qubit.py
