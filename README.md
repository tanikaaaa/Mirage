# Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning

Implementation and reproduction of the CVPR 2025 paper:

**Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning**

Official Mirage Repository:  
https://github.com/NUAA-SmartSensing/Mirage

Our Project Repository:  
https://github.com/tanikaaaa/Mirage

Project Presentation (PPT):  
https://drive.google.com/file/d/1ldUHSbUVGsi6ccXtwK88ui-Cko0k6Uyo/view?usp=drive_link

---

# Overview

Federated Learning (FL) allows multiple clients to collaboratively train a shared global model without sharing raw local data. While privacy-preserving, FL is vulnerable to backdoor attacks where malicious participants poison local updates to implant hidden malicious behavior into the global model.

Most prior work assumes the **Single-Label Backdoor Attack (SBA)** setting, where all attackers target the same class collaboratively.

This project implements and reproduces **Mirage**, the first framework designed for the **Multi-Label Backdoor Attack (MBA)** setting, where:

- multiple independent attackers exist,
- attackers do not communicate,
- attackers target different classes,
- and attacks must survive simultaneously during federated aggregation.

---

# Core Idea of Mirage

Traditional backdoor attacks create **Out-of-Distribution (OOD)** mappings. When multiple attackers independently optimize different target classes, their poisoned samples compete for overlapping neural activation pathways. As a result, only the dominant attacker survives.

Mirage resolves this by constructing:

## In-Distribution (ID) Backdoor Mappings

Instead of creating separate OOD pathways, poisoned samples are optimized to follow the same clean activation pathway as legitimate samples of the target class.

This eliminates inter-attacker competition and allows multiple attackers to coexist without coordination.

---

# Key Contributions of This Project

- Reproduced the Mirage framework using PyTorch.
- Implemented federated learning with multiple independent attackers.
- Reproduced experiments on:
  - CIFAR-10
  - CIFAR-100
  - GTSRB
- Validated Mirage against multiple baseline attacks and defenses.
- Proposed a novelty extension:
  
# Adaptive Frequency-Domain Mirage (AFDM)

AFDM extends Mirage by optimizing triggers in the frequency domain using DCT-based spectral perturbations instead of pixel-space trigger patches.

The reported experimental results in this repository correspond to the original Mirage framework. AFDM is currently implemented as a research-level extension, and full-scale evaluation remains future work. The expected goal of AFDM is to further improve stealthiness, persistence, and robustness against spatial anomaly detection.

---

# Datasets

| Dataset | Classes | Model |
|---|---|---|
| CIFAR-10 | 10 | ResNet-18 |
| CIFAR-100 | 100 | ResNet-18 |
| GTSRB | 43 | ResNet-18 |

---

# Experimental Setup

## CIFAR-10
- Used authors’ pretrained 2000-round checkpoint
- Attack phase: 100 rounds

## CIFAR-100
- Partial pretraining (~80 rounds)
- Attack phase: ~20 rounds

## GTSRB
- Partial pretraining (~90 rounds)
- Attack phase: ~20 rounds

Hardware limitations prevented full 2000-round pretraining for CIFAR-100 and GTSRB.

---

# Main Results (Mirage)

| Dataset | Accuracy (ACC) | Attack Success Rate (ASR) |
|---|---|---|
| CIFAR-10 | 92.25% | 99.84% |
| CIFAR-100 | 64–65% | 29–42% |
| GTSRB | 95.17% | 64.22% |

### CIFAR-10 Per-Attacker ASR
- Attacker 0: 99.93%
- Attacker 1: 99.97%
- Attacker 2: 99.63%

The CIFAR-10 results closely reproduce the paper’s reported performance.

---

# Implemented Attacks

The project compares Mirage against multiple baseline methods:

- Vanilla
- PGD
- Neurotoxin
- Chameleon
- A3FL

---

# Evaluated Defenses

- BackdoorIndicator
- FoolsGold
- Multi-Krum
- FLAME
- DeepSight

Results confirm that existing SBA-oriented defenses are significantly weaker in the MBA setting.

---

# Proposed Novelty: AFDM

## Motivation

Mirage optimizes triggers directly in pixel space, which can introduce spatially visible perturbations.

AFDM moves trigger optimization into the frequency domain using Discrete Cosine Transform (DCT), allowing perturbations to be distributed spectrally rather than spatially.

This potentially improves:
- stealthiness,
- robustness,
- persistence,
- and resistance against spatial anomaly detectors.

---

# AFDM Pipeline

```text
Image
↓
DCT Transform
↓
Spectral Coefficient Optimization
↓
IDCT Reconstruction
↓
Poisoned Image
```

---

# AFDM Features

- Frequency-domain trigger optimization
- Spectral attention mechanism
- Adaptive frequency modulation
- Spectral smoothness regularization
- Differentiable DCT/IDCT pipeline
- Spectral alignment loss

---

# Project Structure

```text
Mirage/
│
├── main.py
├── participants/
├── models/
├── utils/
├── yamls/
├── saved_logs/
├── data/
└── README.md
```

---

# Installation

## Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
pip install torch-dct
pip install pot
```

---

# Running Experiments

## CIFAR-10 Attack

```bash
python3 main.py --params yamls/Mirage/Mirage_cifar10_attack.yaml --dataset CIFAR10 --model_type ResNet18 --no_of_adversaries 3
```

## CIFAR-100 Attack

```bash
python3 main.py --params yamls/Mirage/Mirage_cifar100_attack.yaml --dataset CIFAR100 --model_type ResNet18 --no_of_adversaries 3
```

## GTSRB Attack

```bash
python3 main.py --params yamls/Mirage/Mirage_gtsrb_attack.yaml --dataset GTSRB --model_type ResNet18 --no_of_adversaries 1
```

---

# Technical Challenges

- Missing undocumented YAML parameters
- Apple Silicon MPS backend incompatibilities
- Long FL training runtimes
- Checkpoint resume issues
- Dependency conflicts with `torch_dct` and `POT`

---

# References

1. Ye Li, Yanchao Zhao, Chengcheng Zhu, and Jiale Zhang.  
   *Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning.*  
   CVPR 2025.

2. Mirage Official Repository  
   https://github.com/NUAA-SmartSensing/Mirage

3. Our Implementation Repository  
   https://github.com/tanikaaaa/Mirage
