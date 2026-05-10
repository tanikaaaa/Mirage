# Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning

Implementation and reproduction of the CVPR 2025 paper:

**Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning**

Official Mirage Repository: https://github.com/NUAA-SmartSensing/Mirage

Our Project Repository: https://github.com/tanikaaaa/Mirage

Project Presentation (PPT): https://drive.google.com/file/d/1ldUHSbUVGsi6ccXtwK88ui-Cko0k6Uyo/view?usp=drive_link

Project Report:
https://drive.google.com/file/d/18PTSAG0lrmiFHCzcOmbI6t1yeF9nXyCR/view?usp=sharing

---

# Overview

Federated Learning (FL) enables multiple distributed clients to collaboratively train a shared global model without exposing raw local data. While FL improves privacy, it also introduces security vulnerabilities where malicious participants can poison local model updates and implant hidden backdoor behaviors into the global model.

Most prior research focuses on the **Single-Label Backdoor Attack (SBA)** setting, where all attackers target the same class collaboratively.

This project implements and reproduces **Mirage**, the first framework designed for the **Multi-Label Backdoor Attack (MBA)** setting, where:

- multiple independent attackers exist,
- attackers do not communicate,
- attackers target different classes,
- and all attacks must survive simultaneously during federated aggregation.

---

# Core Idea of Mirage

Traditional backdoor attacks create **Out-of-Distribution (OOD)** mappings.

When multiple attackers optimize different target classes simultaneously, their poisoned samples compete for overlapping neural activation pathways. As a result, only the strongest attacker survives while weaker attacks are suppressed.

The Mirage paper refers to this phenomenon as:

# Infighting

Mirage resolves this problem using:

# In-Distribution (ID) Backdoor Mappings

Instead of forcing poisoned samples through abnormal activation pathways, Mirage aligns poisoned samples with the clean feature-space distribution of the target class.

This allows:
- multiple attackers to coexist,
- simultaneous attack persistence,
- reduced inter-attacker competition,
- and improved resistance against OOD-based defenses.

---

# Project Contributions

This project includes:

- Full implementation and reproduction of Mirage using PyTorch
- Federated learning simulation with multiple attackers
- Experiments on:
  - CIFAR-10
  - CIFAR-100
  - GTSRB
- Comparison against baseline attacks from the paper
- Analysis of existing FL defense limitations
- Proposed novelty extension: **Adaptive Frequency-Domain Mirage (AFDM)**

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

Due to hardware limitations on Apple Silicon MPS backend, full 2000-round pretraining was not feasible for CIFAR-100 and GTSRB.

---

# Main Results (Mirage)

| Dataset | Accuracy (ACC) | Attack Success Rate (ASR) |
|---|---|---|
| CIFAR-10 | 92.25% | 99.84% |
| CIFAR-100 | 64–65% | 13–29% |
| GTSRB | 95.17% | 64.22% |

### CIFAR-10 Per-Attacker ASR
- Attacker 0: 99.93%
- Attacker 1: 99.97%
- Attacker 2: 99.63%

The CIFAR-10 reproduction closely matches the original paper’s reported results, validating correct implementation of the Mirage framework.

---

# Baseline Attacks Studied

- Vanilla
- PGD
- Neurotoxin
- Chameleon
- A3FL

The implementation confirms the exclusion problem in traditional SBA attacks under MBA settings.

---

# Defense Analysis

The project studies Mirage against several federated learning defenses discussed in the original paper, including:

- BackdoorIndicator
- FoolsGold
- Multi-Krum
- FLAME
- DeepSight

The analysis highlights that many defenses designed for SBA settings become significantly weaker in the MBA setting, particularly against Mirage’s In-Distribution (ID) mappings.

---

# Proposed Novelty: Adaptive Frequency-Domain Mirage (AFDM)

## Motivation

The original Mirage framework optimizes triggers directly in pixel space using PGD-based perturbations. Although effective, pixel-space triggers can introduce spatially concentrated perturbations that may become detectable through:
- spatial anomaly analysis,
- perceptual consistency checks,
- or image-quality metrics.

Modern CNNs naturally learn hierarchical spectral representations. Early convolutional layers respond strongly to:
- edges,
- textures,
- frequency patterns,
- and spectral correlations.

This motivates moving trigger optimization from the spatial domain into the frequency domain.

AFDM extends Mirage by representing triggers as frequency-domain perturbations instead of explicit pixel-space trigger patches.

---

# AFDM Core Pipeline

```text
Image
↓
DCT Transform
↓
Frequency-Domain Trigger Injection
↓
Spectral Optimization
↓
Inverse DCT (IDCT)
↓
Poisoned Image
```

Instead of optimizing visible spatial perturbations, AFDM optimizes learnable spectral coefficients using differentiable DCT/IDCT operations.

The original Mirage objectives are preserved:
- adversarial adaptation,
- In-Distribution mapping construction,
- constrained optimization,
- and feature-space persistence.

However, optimization now occurs directly in spectral space.

---

# AFDM Components

## 1. Spectral Attention Mechanism

AFDM identifies important frequency regions and concentrates perturbation energy on high-impact spectral bands while reducing unnecessary low-frequency distortion.

---

## 2. Adaptive Frequency Modulation

Trigger intensity dynamically changes based on current ID mapping quality.

Weak feature alignment:
- stronger perturbation

Stable feature alignment:
- reduced perturbation for improved stealth

---

## 3. Spectral Smoothness Loss

AFDM introduces smoothness regularization to reduce abrupt spectral discontinuities and improve stealthiness against anomaly detectors.

---

# Expected Advantages of AFDM

AFDM is expected to improve:
- stealthiness,
- persistence,
- robustness,
- spectral consistency,
- and resistance against spatial anomaly detectors.

Combining:
- Mirage’s In-Distribution mapping,
with:
- frequency-domain trigger optimization,

creates a potential two-layer evasion strategy against:
1. feature-space anomaly detection,
2. spatial anomaly detection.

---

# Current AFDM Status

The AFDM infrastructure was implemented using:
- `torch_dct`
- differentiable DCT/IDCT operations
- PyTorch autograd
- modified trigger optimization loops

However, full-scale AFDM federated experiments were not completed within the project timeline due to computational overhead introduced by repeated DCT/IDCT operations during trigger optimization.

Therefore:
- all reported experimental results correspond to the original Mirage framework,
- while AFDM remains a research-level novelty extension and future work direction.

---

# Folder Structure

```text
Mirage/
│
├── Mirage/
│   ├── data/
│   ├── datasets/
│   ├── models/
│   ├── participants/
│   ├── saved_logs/
│   ├── saved_models/
│   ├── utils/
│   ├── yamls/
│   │
│   ├── main.py
│   ├── extract_metrics.py
│   ├── plot_metrics.py
│   ├── metrics.csv
│   ├── accuracy.pdf
│   ├── asr.pdf
│   └── README.md
│
├── venv/
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

# References

1. Ye Li, Yanchao Zhao, Chengcheng Zhu, and Jiale Zhang.  
   *Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning.*  
   CVPR 2025.

2. Mirage Official Repository  
   https://github.com/NUAA-SmartSensing/Mirage

3. Our Implementation Repository  
   https://github.com/tanikaaaa/Mirage
