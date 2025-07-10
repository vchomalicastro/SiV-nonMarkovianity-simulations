# SiV-nonMarkovianity-simulations
Numerical simulations of phonon-induced non-Markovianity in silicon-vacancy (SiV) centers in diamond, as used in "Phonon-induced non-equilibrium dynamics of a single solid-state spin" (Norambuena, Tancara, Chomali-Castro, Castillo, 2024).

---

## Overview

This repository contains Python code for simulating memory effects and non-Markovian dynamics in a single solid-state spin (the negatively charged silicon-vacancy SiV– center in diamond).  
Specifically, this code generates the results for the non-Markovianity heatmaps in Section 3.2 (and Appendix D) of the above paper.

## Paper

- **Title:** Phonon-induced non-equilibrium dynamics of a single solid-state spin
- **Authors:** Ariel Norambuena, Diego Tancara, Vicente Chomali-Castro, Daniel Castillo
- **Year:** 2024
- **arXiv:** [arxiv.org/abs/2411.09825](arxiv.org/abs/2411.09825)

## Repository Structure

- `NBLP_nonHPC_version.py` — Adapted simulation code for non-supercomputer environments.
- `heatmap_generator_Bz_0_2e1_Bx_0_2e1_builtin_data.py` — Heatmap generator script with built-in data for the range [0, 2e1 T] (Figure 4(a)).
- `heatmap_generator_Bz_0_2e2_Bx_0_2e2_builtin_data.py` — Heatmap generator script with built-in data for the range [0, 2e2 T] (Figure 4(b)).
- `README.txt` — This file.
- `LICENSE` — MIT License text.

## About the Code and Data

The original simulations for Section 3.2 of the paper (with the computational implementation explained in Appendix D) were run in parallel across 40 separate scripts (e.g., `1.py`, `2.py`, ..., `40.py`), each covering a different region of the $(B_x, B_z)$ parameter space to compute the full non-Markovianity $\mathcal{N}_{\rm BLP}(B_x, B_z)$ heatmap.

This repository includes:
- An **adapted version** of the simulation code (`NBLP_nonHPC_version.py`) for running on standard desktop/laptop environments (locally parallelized through `joblib`, though not split into multiple files).
- **Heatmap generator scripts** with **built-in data** (`heatmap_generator_Bz_0_2e1_Bx_0_2e1_builtin_data.py` and `heatmap_generator_Bz_0_2e2_Bx_0_2e2_builtin_data.py`).  
  These scripts use the actual data produced by our simulations run on the Illinois Campus Cluster high-performance environment, under the Illinois Computes program, as reported in the paper. This allows users to regenerate the heatmaps directly or analyze the data in general.

All necessary data for reproducing the published figures is included, so users do **not** need access to a supercomputer to reproduce the main results.

## Installation

This code requires Python 3.8+ and the following packages:

- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `qutip`
- `joblib`

Install dependencies with:

    pip install numpy scipy matplotlib seaborn qutip joblib

Or with conda:

    conda install numpy scipy matplotlib seaborn qutip joblib

