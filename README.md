# Mirage Backdoor Attack on Federated Learning (CIFAR-100)

## Overview
This project implements a **backdoor attack in Federated Learning (FL)** using the **Mirage attack algorithm**. The experiments are conducted on image classification datasets such as **CIFAR-10** and **CIFAR-100** using a **ResNet18 model**.

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

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- YAML
- tqdm

Install dependencies:

```bash
pip install torch torchvision numpy pyyaml tqdm colorama torchsummary

