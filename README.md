# Mirage Backdoor Attack on Federated Learning (CIFAR-100)

## Overview
TThis project implements a backdoor attack in Federated Learning (FL) using the Mirage attack algorithm, inspired by recent research on multi-label backdoor attacks in federated settings [1], and evaluates the trade-off between model accuracy (ACC) and attack success rate (ASR) under varying adversarial conditions.
The goal is to analyze how adversarial clients can manipulate a global model while maintaining high performance on clean data.

---

## Objectives
- Study the impact of **malicious clients in federated learning**
- Evaluate the trade-off between:
  - **Accuracy (ACC)** on clean data  
  - **Attack Success Rate (ASR)** on poisoned inputs  
- Understand how attack strength affects model robustness  

---

## Key Concepts

### Federated Learning
A decentralized learning setup where:
- Multiple clients train locally
- A central server aggregates updates
- Raw data is never shared

### Backdoor Attack
A malicious strategy where:
- A trigger pattern is inserted into inputs
- Model learns to misclassify triggered inputs into a target label

### Mirage Attack
An advanced backdoor attack that:
- Learns optimized triggers
- Uses discriminator-based training
- Maintains stealth while achieving high ASR

---

## Features
- Multi-client federated training  
- Support for benign and malicious participants  
- Configurable adversarial settings  
- Dynamic trigger optimization  
- Compatible with multiple datasets  

---

## Training Pipeline

1. Initialize global model  
2. Distribute model to selected clients  
3. Each client performs local training  
4. Malicious clients inject poisoned updates  
5. Server aggregates updates (FedAvg)  
6. Evaluate global model (ACC & ASR)  
7. Repeat for multiple rounds

---

## Attack Pipeline (Mirage)

1. Malicious client receives global model  
2. Optimizes trigger using gradient-based search  
3. Trains discriminator to refine trigger quality  
4. Injects poisoned samples into local training  
5. Generates malicious model update  
6. Sends update to server for aggregation

---

## Key Insights

- Increasing adversaries significantly improves ASR but reduces model accuracy  
- Larger trigger optimization improves attack strength but increases runtime  
- CIFAR-100 is more challenging than CIFAR-10 due to higher class diversity  
- Balanced configurations achieve high ASR without severely degrading accuracy

---

## Reproducibility

- Random seed fixed for consistent results  
- All configurations defined in YAML files  
- Experiments can be reproduced using provided scripts

---

## Limitations

- High computational cost during attack phase  
- Performance depends on hyperparameter tuning  
- Limited to small-scale datasets (CIFAR-10/100)

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- YAML
- tqdm

---

## References

[1] Ye Li, Yanchao Zhao, Chengcheng Zhu, Jiale Zhang  
**Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning**  
Proceedings of CVPR 2025  
https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Infighting_in_the_Dark_Multi-Label_Backdoor_Attack_in_Federated_Learning_CVPR_2025_paper.pdf  
