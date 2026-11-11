# Model Card -- PredictMed CKD Deterioration (research prototype)

> Research estimate only. Not a diagnosis, not a treatment recommendation, not validated for clinical use.

## Intended use
Rank health-plan members by estimated probability of **recorded kidney deterioration within 12 months** (>=40% sustained eGFR decline, eGFR<15, or dialysis start) to support *human* care-management outreach under a fixed capacity. Outreach decisions remain with clinicians and care managers (HIL approval queue).

## Data
- Synthetic, literature-calibrated payer cohort (50,000 members, 2019-2025): labs, claims, pharmacy, enrollment, area-level SDOH. **Synthetic results do not demonstrate real-world performance.**
- Cohort version `cohort-ckd-payer-v1`; snapshot schema `features.snapshot.v1`; knowledge base `kdigo2024-ckdepi2021-kfre4-v1`; twin `twin-llt-robust-v1`.
- Real-data check: UCI Diabetes 130-US Hospitals (separate 30-day readmission study, see reports/uci_report.json).

## Evaluation design
Member-grouped **and** out-of-time: train members/index dates <= 2022-01; validation 2022-07..2023-01 on different members; test 2023-07..2024-01 on a third, disjoint member set. Calibration, thresholds, conformal and gates used validation only; the test split was scored once after the bundle hash was locked.

## Locked test performance
| Metric | Value | 95% CI (member bootstrap) |
|---|---|---|
| AUROC | 0.897 | 0.876-0.915 |
| PR-AUC | 0.589 (prevalence 0.059) | 0.526-0.648 |
| Brier | 0.0348 | 0.0301-0.0393 |
| Calibration slope / intercept | 1.01 / -0.04 | |
| Top-10% capture | 67.5% | 63.2%-72.7% |

Baselines on the same test rows: kfre_equation(recalibrated) AUROC 0.868 / PR-AUC 0.392; logistic AUROC 0.896 / PR-AUC 0.551; prevalence AUROC 0.500 / PR-AUC 0.059

## Fairness
Race/ethnicity is excluded from predictors and used only for auditing. Subgroup AUROC, observed/expected ratio and true-positive rate at the capacity threshold are in `reports/subgroups_test.csv`. Excluding a sensitive attribute does not guarantee fairness: area-level SDOH and utilisation can act as proxies, and treatment under-use correlated with deprivation is present in the data-generating process by design.

## Known limitations
- Outcome is *recorded* deterioration; members with sparse labs have under-ascertained outcomes (quantified against simulator ground truth in the ascertainment section of the test report).
- Members who die or disenroll before 12 months without an event are excluded (censoring).
- Intervention relative risks used for prioritisation are literature priors, not effects estimated here.
- Crash-dialysis model has few events; treat as exploratory.

## Promotion
Gates: {"auprc_lift": true, "auroc": true, "calibration_slope": true, "ece": true, "subgroup_auroc_gap": true, "conformal_coverage": true}. Approval: approved (model_promotion-439b3d8a8a, demo auto-approval).