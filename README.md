# FJSP-LB-MK  
Flexible Job Shop Scheduling with Limited Buffers and Material Kitting

Official implementation of the FJSP-LB-MK framework.

---

## Overview

FJSP-LB-MK extends the classical Flexible Job Shop Scheduling Problem (FJSP) by incorporating realistic industrial constraints:

- Limited buffer containers  
- Material kitting constraints  
- Pallet replacement delays  
- Category-based part sorting  

The problem is motivated by real steel plate production lines. After cutting operations, steel plates are decomposed into part groups that must be sorted into limited-capacity buffer containers. When a container becomes full, pallet replacement and transportation time are incurred. Part groups from different jobs sharing the same category are encouraged to use the same buffer container, introducing inter-job coupling constraints.

### Objective

Minimize:

- Makespan  
- Pallet replacement frequency  
- Sorting inefficiency  

---

## Repository Structure

FJSP-LB-MK/
│
├── data/ # Synthetic and real datasets
├── env/ # Scheduling environment
├── models/ # DRL models (ours, HGNN, DAN)
├── baselines/ # OR-Tools, PDRs, etc.
├── trainer/ # PPO training framework
├── evaluation/ # Evaluation scripts
├── ablation/ # Ablation studies
├── simulation/ # Visualization tools
├── configs/ # Experiment configurations
├── scripts/ # Reproduction scripts
└── README.md


---

## Dataset

### Synthetic Dataset

Multiple scales:

- 10×5  
- 20×5  
- 20×10  
- 30×10  
- 40×10  

Each scale includes training, validation, and test instances.

### Real Production Dataset

Collected from four industrial production lines:

- Line A  
- Line B  
- Line C  
- Line D  

Each contains training, validation, and test sets.

---

## Installation

```bash
conda create -n fjsp python=3.9
conda activate fjsp
pip install -r requirements.txt

## Training
## Evaluation
## Baselines

Implemented baselines include:

OR-Tools CP-SAT

Priority Dispatching Rules (FIFO, SPT, MWR, etc.)

DAN

HGNN

All methods share a unified environment interface to ensure fair comparison and reproducibility.
