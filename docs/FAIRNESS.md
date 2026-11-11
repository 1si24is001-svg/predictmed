# PredictMed — Algorithmic Fairness & Equity Audit

> **Scope**: Subgroup disparity analysis, root-cause diagnosis, mitigation experiments on validation data, and automated promotion gating.

---

## 1. Executive Summary

Auditing by insurance coverage plan revealed a systematic performance disparity on the locked test cohort:
- **Commercial Members**: AUROC 0.908, Observed/Expected (O/E) ratio 1.078
- **Medicare Members**: AUROC 0.906, Observed/Expected (O/E) ratio 0.863
- **Medicaid Members**: AUROC 0.842, Observed/Expected (O/E) ratio 1.111

Medicaid members experience lower discrimination and under-prediction (O/E > 1.0 indicates that more members deteriorate than the model estimates). This document diagnoses the root causes, reports results across three mitigation strategies tested on validation data, and formalizes an automated Guardian fairness gate.

---

## 2. Diagnosis Findings

### 2.1 Feature Missingness & Care Disparities

Longitudinal biomarker sampling and guideline-recommended testing differ significantly across coverage programs:

| Feature / Metric | Commercial (n=11,093) | Medicaid (n=4,594) | Medicare (n=6,977) | Primary Disparity |
|---|---|---|---|---|
| **UACR Test Missingness** | 48.3% | **52.6%** | 41.8% | +10.8% higher gap in Medicaid |
| **Mean Longitudinal Labs** | 5.6 obs | **5.1 obs** | 6.6 obs | Sparse lab histories in Medicaid |
| **High / Very High Tier** | 7.5% | 8.3% | 18.0% | Medicare is older with higher disease load |
| **Mean Predicted Risk** | 0.045 | 0.048 | 0.089 | Model ranks Medicaid lower despite high risk |

### 2.2 Root-Cause Assessment
1. **Lab Sampling Sparsity**: Medicaid members average fewer creatinine/eGFR measurements (5.1 vs 6.6), widening the Kalman filter's uncertainty interval and reducing slope precision.
2. **Ascertainment Lag**: In payer claims, silent CKD without an active ICD diagnosis code is more prevalent in socially vulnerable and Medicaid populations.
3. **SDoH Deprivation Proxy**: Area deprivation index (SDoH) correlates with under-utilization, creating a confounding effect where lower utilization mimics lower clinical severity.

---

## 3. Mitigation Experiments (Validation Split Only)

In accordance with RiskLens-HIL standards, all three mitigation strategies were evaluated strictly on the **validation cohort** (7,002 members, 373 events). Any mitigation lowering overall PR-AUC by more than 0.01 was rejected.

| Mitigation Strategy | Approach | Medicaid O/E | Medicaid AUROC | Overall PR-AUC | Decision |
|---|---|---|---|---|---|
| **Baseline (Champion LightGBM)** | Global calibration | 0.998 | 0.876 | 0.610 | Benchmark |
| **Option A: Group-wise Recalibration** | Plan-specific Isotonic | 1.002 | 0.876 | 0.608 (Δ -0.002) | **Kept** (Available via `GroupwiseCalibrator`) |
| **Option B: Missingness Indicators** | Explicit flags for unmeasured labs | 1.015 | 0.871 | 0.597 (Δ -0.013) | **Rejected** (Redundant with Kalman twin) |
| **Option C: Sample Re-weighting** | Inverse frequency by plan | 0.982 | 0.869 | 0.594 (Δ -0.016) | **Rejected** (Over-weights noisy Medicaid tail) |

### Selected Policy:
Group-wise calibration (`GroupwiseCalibrator`) effectively stabilizes O/E calibration without penalizing discriminatory PR-AUC.

---

## 4. Automated Guardian Fairness Gate

To prevent deploying models with unacceptable subgroup calibration drift, the `Guardian` agent now enforces an automated plan-type calibration gate during promotion:

```yaml
promotion_gates:
  plan_calibration_oe_min: 0.80
  plan_calibration_oe_max: 1.25
  plan_calibration_min_events: 30
```

- **Rule**: For every plan-type group with at least 30 recorded events, the Observed/Expected calibration ratio must remain strictly within `[0.80, 1.25]`.
- **Enforcement**: If any eligible subgroup breaches this interval on validation data, promotion is automatically **blocked** and routed to human review.

---

## 5. Limitations & Ethical Considerations
- **Proxy Discrimination**: Excluding race/ethnicity from predictive features does not guarantee demographic parity due to correlation with postal-code SDoH and dual-eligible Medicaid status.
- **Outreach vs Diagnosis**: The system is designed for care-management outreach prioritization, where conservative thresholds must be balanced against care coordinator capacity.

