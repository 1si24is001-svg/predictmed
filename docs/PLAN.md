# PredictMed — Technical Plan

**Problem statement 2 (HIL):** Chronic condition deterioration detection from a payer context.
PredictMed is an agentic AI system for chronic kidney disease (CKD) that creates value for health plans
across member management, utilization, medical cost, quality and value-based programmes.

This plan applies the RiskLens-HIL engineering standard to the CKD problem. That standard covers leakage
controls, member-grouped splits, calibration, subgroup audits, Kafka contracts with a DLQ and idempotency, and
a model card. Five additions shape the design around how payer data actually behaves.

---

## 1. Prediction problem

| Item | Definition |
|---|---|
| Unit | Health-plan member at an as-of **index date** T (rolling, every 6 months) |
| Eligibility | Age ≥18; enrolled ≥12 months before T; alive, enrolled and not on dialysis at T; ≥1 outpatient eGFR in [T−24m, T) with last value ≥15 |
| **Primary target** | Recorded kidney deterioration in [T, T+12m): dialysis start, **or** an outpatient eGFR ≤60% of baseline (the ≥40% decline accepted by FDA/EMA as a surrogate endpoint), **or** eGFR <15. Either eGFR event counts unless a later value >70% of baseline within 90 days refutes it. Inpatient (acute) values never define the outcome. |
| Secondary target | **Crash dialysis** in [T, T+24m): first dialysis claim in an inpatient or ED setting with no nephrology claim in the prior 180 days. This mirrors the AIR/CMS core definition; about 1 in 3 Medicare beneficiaries who need dialysis start this way. |
| Prediction-time contract | Every feature uses rows dated < T only. This is enforced by a truncation-invariance test. |
| Censoring | Members who die or disenroll before T+12m without an event are excluded. This is documented. |

## 2. Data strategy (hybrid)

| Source | Role | Claim it supports |
|---|---|---|
| **Synthetic payer cohort** (`simulator.py`, 50k members, 2019–2025) | Longitudinal labs, claims, pharmacy, enrollment and area-level SDOH. Calibrated to CKD-EPI 2021, KDIGO staging, SGLT2i RR 0.63 and KDIGO rapid progression (>5 mL/min/yr). | Pipeline correctness and *relative* comparisons of methods. **It does not show real-world performance.** |
| **UCI Diabetes 130-US Hospitals** (real) | Separate 30-day readmission study with a diabetic-kidney-disease subgroup | An honest real-data benchmark under the RiskLens protocol |
| Synthea / CMS SynPUF / MIMIC-IV | Adapter targets (event envelope is source-agnostic) | Future integration; never concatenated |

The simulator deliberately reproduces the problems found in real payer data:

- informative, irregular lab sampling;
- creatinine noise, differences between lab vendors, and transient pre-renal dips;
- inpatient AKI spikes;
- under-coding of CKD (silent CKD);
- treatment under-use that tracks deprivation;
- a hidden rapid-progressor phenotype, plus *unpredictable onset* of accelerated decline;
- death as a competing risk, and disenrollment.

Ground truth is written to an **oracle** table that no feature code may read.

## 3. Architecture

```
Sources ─► Kafka topics (envelope v1) ─► state-writer (validate │ DLQ │ idempotent SQLite state │ online twin)
   │                                        └► scheduler tick (as-of) ─► features.snapshot.v1 ─► risk.score.request.v1
   │                                                                     scorer (bundle, retries) ─► risk.score.result.v1 ─► audit-sink
   └► Airflow batch ─► Iceberg bronze ─► silver (validated) ─► gold.patient_timeline / gold.feature_snapshots
                                   │ (time travel: run manifests pin snapshot ids)
                                   ▼
          FeatureScout ─► ModelSmith (MLflow, Slurm) ─► Guardian (gates → HIL approval → locked test) ─► bundle
                                                                                    │
                                     FastAPI  ◄───────────────────────────────────────┤
                                     Streamlit ◄── CareNavigator (gaps, EAC, worklist, briefs)
```

## 4. What makes it different

1. **Kidney digital twin.** For each member, a robust Bayesian local-linear-trend Kalman filter tracks eGFR. Its prior is fitted by empirical Bayes on training members only. Inpatient labs get higher measurement noise, and Huber-style inflation absorbs outliers. It outputs a denoised eGFR, a slope with uncertainty, *P(rapid progression)*, a **kidney runway** (months until eGFR 15) and *P(eGFR<15 within 24m)*. The same class runs in batch and on each Kafka lab event, so the two paths match by construction.
2. **Missing data as a signal.** Gaps in care are encoded as features: no UACR test, no recent eGFR, no nephrology visit, and **silent CKD** (eGFR persistently <60 with no N18 code).
3. **Uncertainty with a human in the loop.**
   - Platt or isotonic calibration, chosen on disjoint validation halves.
   - **Mondrian conformal** prediction sets give coverage guarantees for each class.
   - **Venn-Abers** gives a probability interval for each member.
   - Members with "uncertain" prediction sets go to human review first.
4. **Expected avoidable cost.** The worklist ranks members by
   `EAC = P(det)·(1−Π RR_gap)·C_prog + P(crash)·(1−RR_referral)·C_crash`.
   The RR values come from trials (SMART-C SGLT2i 0.63, RENAAL/IDNT, FIDELIO-DKD). This is tested with an
   **in-silico randomized outreach trial**: the simulator re-runs with common random numbers, so the only
   difference from the factual world is the causal effect of the intervention.
5. **Agent team.** Agents use typed tools and write evidence to a journal. They *propose*, and humans approve:
   - **DataSentinel:** schema checks, plausibility checks and PSI drift.
   - **FeatureScout:** the autonomous feature-selection agent. It applies the policy gate, then a leakage
     tribunal (truncation probe plus a "too good to be true" check), removes redundant features, runs a
     cost-aware forward group search, prunes unstable features using permutation stability, and finishes with
     a fairness guard.
   - **ModelSmith:** runs the candidate zoo, tunes with grouped CV, calibrates, fits conformal sets and logs
     to MLflow.
   - **Guardian:** applies promotion gates, sends the model to HIL approval, locks the bundle by hash, runs the
     one-time test and writes the model card.
   - **CareNavigator:** finds care gaps, computes EAC, builds the capacity-limited worklist and writes outreach
     briefs. An optional Claude model can narrate the briefs.

## 5. Evaluation protocol

- **Splits are member-grouped and out-of-time.**
  - Train: train members, index dates ≤ 2022-01.
  - Valid: other members, 2022-07 to 2023-01.
  - Test: a third member set, 2023-07 to 2024-01.
  - The twin prior is fitted on data before 2023-01-01.
- **Candidates:**
  - prevalence;
  - **KFRE 4-variable equation** (the clinical baseline to beat);
  - logistic regression;
  - random forest;
  - MLP (the deep-learning comparison);
  - CatBoost;
  - LightGBM.
- **Metrics:**
  - AUROC and PR-AUC, with prevalence reported alongside;
  - Brier score, ECE, and calibration slope and intercept;
  - top 5%, 10% and 20% capture, PPV and lift;
  - sensitivity and specificity at the capacity threshold;
  - member-bootstrap 95% CIs;
  - subgroup tables;
  - the subpopulation with eGFR<60;
  - **ascertainment gap** against the oracle;
  - conformal coverage.
- **Gates (validation only):**
  - PR-AUC lift ≥2;
  - AUROC ≥0.75;
  - calibration slope between 0.8 and 1.2;
  - ECE ≤0.03;
  - race AUROC gap ≤0.08, among groups with ≥30 events;
  - conformal coverage.

## 6. Phases (as on the HIL slide)

| Phase | Deliverable in this repo |
|---|---|
| 1 · Discovery & data assessment | `simulator.py`, `cohort.py`, `uci_track.py`, `data_manifest/`, DataSentinel audit |
| 2 · Pipeline & feature store | `streaming/` (Kafka contracts, DLQ, idempotency), `dags/` (Airflow), `lakehouse/` (Iceberg), `features/` (point-in-time snapshots, twin, lab trends) |
| 3 · Agentic AI development | `agents/` (5 agents + HIL queue), `models/` (zoo, calibration, conformal, SHAP), MLflow tracking, `slurm/` |
| Serving | `serving/api.py` (FastAPI), `app/streamlit_app.py` |

## 7. Out of scope (v1)

- Clinical deployment.
- Autonomous outreach (every action goes through HIL).
- RL policies.
- Claims of real-world validity from synthetic data.
- Joint training across incompatible sources.
