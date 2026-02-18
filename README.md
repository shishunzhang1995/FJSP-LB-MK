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
