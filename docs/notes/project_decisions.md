# Project Decisions

## 2026-06-XX

### Context

Following supervisor review of Proposal v1, the project scope was refined and upgraded from a simple XAI evaluation study into a reliability-aware XAI framework for chest X-ray diagnosis.

---

## Decision 1

### Title Direction

Move toward:

Reliability-Aware Explainable AI for Chest X-ray Diagnosis

instead of a simple XAI comparison study.

---

## Decision 2

### Core Evaluation Dimensions

The framework will evaluate explanations across five dimensions:

- Localization
- Faithfulness
- Stability
- Anatomical Plausibility
- Radiologist Agreement

---

## Decision 3

### Reliability Profile

Introduce a multidimensional reliability profile for every explanation.

Purpose:

Represent strengths and weaknesses of each XAI method across all evaluation dimensions.

---

## Decision 4

### Composite Explanation Reliability Index (CERI)

Define a unified reliability score derived from the reliability profile.

Purpose:

Enable quantitative comparison and ranking of XAI methods.

---

## Decision 5

### Anatomical Plausibility Score (APS)

Introduce APS to measure whether explanations focus on clinically meaningful anatomical regions.

Purpose:

Detect shortcut learning and anatomically implausible attention patterns.

---

## Decision 6

### Robust Ranking Framework

Develop a statistically grounded ranking strategy for XAI methods.

Purpose:

Avoid rankings based solely on average scores.

---

## Decision 7

### Stability-Enhanced Explanation Averaging (SEEA)

Develop a lightweight explanation aggregation strategy.

Purpose:

Improve stability while preserving localization and faithfulness.

---

## Decision 8

### Dataset Strategy

Primary Dataset:

- CheXpert

Localization Evaluation Datasets:

- VinDr-CXR
- RSNA Pneumonia Detection
- SIIM-ACR Pneumothorax

---

## Decision 9

### Clinical Validation

Include at least two radiologists for limited-scale validation.

Target:

100–300 images.

---

## Decision 10

### Literature Review Structure

Organize prior work according to:

- CAM-based XAI
- Gradient-based XAI
- Perturbation-based XAI
- Attention-based XAI

instead of general deep-learning literature.
