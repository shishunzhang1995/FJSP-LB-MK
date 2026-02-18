# FJSP-LB-MK
**Flexible Job Shop Scheduling with Limited Buffers and Material Kitting**

Official implementation of the FJSP-LB-MK framework. This repository provides a Deep Reinforcement Learning (DRL) solution for complex industrial scheduling problems involving shared resource constraints and material sorting.

---

## Overview

FJSP-LB-MK extends the classical Flexible Job Shop Scheduling Problem (FJSP) by incorporating realistic industrial constraints:

- **Limited Buffer Containers**: A finite number of pallets/containers are available for part sorting.
- **Material Kitting**: Parts from different jobs belonging to the same category are encouraged to share the same buffer.
- **Pallet Replacement Delays**: Switching between different part categories in a full buffer incurs AGV transportation and waiting time.
- **Category-based Part Sorting**: Multi-job coupling through shared material categories.

The problem is motivated by real steel plate production lines where cutting operations are followed by complex sorting and kitting requirements.

---

## Repository Structure

```text
FJSP-LB-MK/
│
├── data/           # Synthetic and real-world industrial datasets
├── env/            # Scheduling environment simulating pallet changes
├── models/         # DRL models (Relational-Graph HGNN, DAN, etc.)
├── baselines/      # OR-Tools, PDRs, etc.
├── trainer/        # PPO training framework
├── evaluation/     # Evaluation scripts
├── ablation/       # Ablation studies
├── simulation/     # Visualization tools
├── configs/        # Experiment configurations
├── scripts/        # Reproduction scripts
└── README.md

## Dataset
### Synthetic Dataset
The synthetic data includes multiple scales to test scalability:

10×5 (Jobs × Machines)

20×5

20×10

30×10

40×10

Each scale includes dedicated training, validation, and test instances.

### Real Production Dataset
Collected from four distinct industrial steel plate production lines:

Line A

Line B

Line C

Line D

Each contains training, validation, and test sets to verify the model's performance in real industrial environments.

## Installation
Ensure you have Conda installed.

Bash

# Create and activate environment
conda create -n fjsp python=3.9
conda activate fjsp

# Install dependencies
pip install -r requirements.txt
## Training & Evaluation
### Training
Train on synthetic dataset:

Bash

bash scripts/train_synthetic.sh
Fine-tune on real dataset:

Bash

bash scripts/finetune_real.sh
### Evaluation
Greedy Evaluation:

Bash

python evaluation/evaluate.py --mode greedy
Sampling Evaluation:

Bash

python evaluation/evaluate.py --mode sampling --num_samples 100
## Baselines
The repository includes the following baseline methods for comparison:

OR-Tools CP-SAT: A state-of-the-art constraint programming solver.

Priority Dispatching Rules (PDRs): Includes FIFO (First-In-First-Out), SPT (Shortest Processing Time), MWR (Most Work Remaining), etc.

DAN: Dual-Attention Network for FJSP.

HGNN: Heterogeneous Graph Neural Network baselines.

All methods share a unified environment interface to ensure fair comparison and reproducibility.

## Ablation Studies
The code supports various ablation configurations:

Remove buffer node: Investigating the impact of the centralized buffer state.

Uniform edge weights: Comparing cost-sensitive weights against standard edges.

Benefit-seeking strategy: Evaluating proactive vs. reactive scheduling.

Lambda sensitivity analysis: Testing the weight of pallet-change penalties in the reward function.

Example:

Bash

python ablation/lambda_sensitivity.py
## Citation
If you use this repository or the FJSP-LB-MK framework in your research, please cite:


@article{zhang2025fjsplbmk,
  title={Learning Flexible Job Shop Scheduling under Limited Buffers and Material Kitting Constraints},
  author={Zhang, Shishun and Xu, Juzhan, and Fan, Yidan, and Zhu, Chenyang, and Hu, Ruizhen, and Wang, Yongjun, and Xu, Kai},
  journal={Advanced Engineering Informatics},
  year={2026}
}
