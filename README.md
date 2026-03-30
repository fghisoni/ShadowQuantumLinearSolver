# Shadow Quantum Linear Solver

Codebase for reproducing the numerical experiments of the Shadow Quantum Linear Solver (SQLS) and its comparison against Hadamard-test VQLS baselines on several quantum linear system problem classes.

## Repository structure

- `Ising/`  
  Ising-inspired QLSP instances and corresponding SQLS / Hadamard-test VQLS scripts.

- `PGLS/`  
  Potential-grid / Poisson-like linear system instances, including input data files (`*.txt`, `*.npy`) and experiment scripts.

- `RQLSP1/`, `RQLSP2/`  
  Randomly generated QLSP benchmarks and comparison scripts.

- `Plots/`  
  Jupyter notebooks and helper files used to generate figures and analyze results.

## Main scripts

Typical experiment files are:

- `SLS_potentials_v2.py` — SQLS-based optimization runs
- `Htest_VQLS_*.py` — Hadamard-test VQLS baselines
- `Ising_Inspired_QLSP.py`, `Random_QLSP_v8.py` — problem instance generation
- `Pauli_algebra_v*.py` — operator-processing utilities

## Requirements

The code relies mainly on:

- Python 3
- PennyLane 0.35.0
- NumPy 1.26.4
- SciPy 1.12.0
