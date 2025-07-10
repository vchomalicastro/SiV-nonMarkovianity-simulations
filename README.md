# SiV-nonMarkovianity-simulations
Numerical simulations of phonon-induced non-Markovianity in silicon-vacancy (SiV) centers in diamond, as used in "Phonon-induced non-equilibrium dynamics of a single solid-state spin" (Norambuena, Tancara, Chomali-Castro, Castillo, 2024).

---

## Overview

This repository contains Python codes for simulating memory effects and non-Markovian dynamics in a single solid-state spin (the negatively charged silicon-vacancy SiV– center in diamond). Specifically, these were the codes used to generate results from the above paper, including both the effective Rabi model (Section 3.1) and the four-level system with transverse field (Section 3.2 and Appendix D).

## Paper

- **Title:** Phonon-induced non-equilibrium dynamics of a single solid-state spin
- **Authors:** Ariel Norambuena, Diego Tancara, Vicente Chomali-Castro, Daniel Castillo
- **Year:** 2024

## Repository Structure

- `blp_measure/` — Code to compute the BLP non-Markovianity measure as explained in Appendix D; this generates the results of Section 3.2.
- `rabi_model/` — Code for the effective Rabi model (Section 3.1), if included.
- `data/` — Example input data and simulation outputs.
- `README.txt` — This file.
- `LICENSE` — MIT License text.

## Installation

This code requires Python 3.8+ and the following packages:

- numpy
- scipy
- matplotlib
- seaborn
- qutip
- joblib

Install dependencies with:

    pip install numpy scipy matplotlib seaborn qutip joblib

Or with conda:

    conda install numpy scipy matplotlib seaborn qutip joblib
