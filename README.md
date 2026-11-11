# PredictMed — Agentic AI for CKD deterioration (payer context)

PredictMed answers problem statement #2 of the HIL brief: *Chronic Condition Deterioration Detection from a Payer
Context — Agentic AI for Chronic Kidney Disease*. It applies the **RiskLens-HIL** engineering standard
throughout: leakage controls, grouped splits, calibration, subgroup audits, Kafka contracts with a DLQ and
idempotency, and a model card.

> Research prototype. It produces estimates to support human care management. Its outputs are not diagnoses or
> treatment recommendations. Results on the synthetic cohort do not demonstrate real-world performance.

## What's different

| # | Idea | Why it matters to a health plan |
|---|---|---|
| 1 | **Kidney digital twin.** A robust Bayesian Kalman filter for each member's eGFR, with an empirical-Bayes prior. It gives a denoised eGFR, a slope with uncertainty, *P(rapid progression)* and a **kidney runway** (months until eGFR 15). | Payer labs are sparse and noisy. Against ground truth, the twin's slope error is **3.3× lower** than a raw regression line, and it still gives a slope for the 40% of members who don't have enough labs to fit one. |
| 2 | **Missing data treated as a signal.** Features include no UACR test, no recent eGFR, no nephrology visit, and **silent CKD** (eGFR persistently <60 with no N18 code). | About 87% of US adults with CKD don't know they have it (CDC). Closing these gaps is work a plan can act on. |
| 3 | **Uncertainty with a human in the loop.** Conformal prediction sets give coverage guarantees for each class, Venn-Abers gives each member a probability interval, and uncertain members are routed to people. | The system says "I'm not sure" instead of guessing. |
| 4 | **Expected avoidable cost (EAC).** Risk × the share of risk that open gaps could fix (trial relative risks: SGLT2i 0.63, RAS inhibitors, finerenone) × cost, plus crash-dialysis avoidance. | It targets members where outreach can change the outcome, not only those at highest risk. |
| 5 | **In-silico randomized outreach trial.** The simulator re-runs with common random numbers, so outcomes change *only* through the intervention's causal effect. | It tests whether EAC targeting averts more **true** deterioration at the same capacity. |
| 6 | **Agent team with guardrails.** DataSentinel, FeatureScout, ModelSmith, Guardian and CareNavigator act through typed tools and record evidence in a journal. A person approves every promotion and outreach. Claude can optionally narrate. | It is autonomous where that is safe and accountable where it matters. |

## Build order

1. **Get CKD data.** The real UCI Chronic Kidney Disease dataset (`ckd_baseline.py`), the literature-calibrated synthetic payer cohort (`simulator.py`) and UCI Diabetes 130.
2. **Understand and clean.** Profiling and dirty-token repair (e.g. `ckd\t` and `\t?`), plausibility checks, a data dictionary, missingness reports and the DataSentinel audit.
3. **Baseline ML models.** Prevalence, logistic regression, random forest, LightGBM and CatBoost, plus the KFRE clinical equation.
4. **Evaluate.** Repeated stratified CV and a held-out split, AUROC, PR-AUC, Brier score, sensitivity/specificity, calibration, CIs and subgroups.
5. **Then the platform.** Agents, Kafka streaming, Apache Iceberg, MLflow, serving and the dashboard.

**Steps 1–4 on real CKD data** (UCI CKD, n = 400, 62.5% CKD; `python -m predictmed.pipeline ckd`):

| Feature set | Logistic AUROC | Random forest AUROC | LightGBM AUROC |
|---|---|---|---|
| All features | 1.000 | 1.000 | 1.000 |
| Non-renal routine data only (no creatinine, urea, albumin, specific gravity or urinalysis) | 0.994 | 0.993 | 0.995 |

These scores are near-perfect because the label is a **diagnosis** made from the same kidney markers the model
sees. They don't show that the model can predict **deterioration**. That question needs longitudinal labs, which
is why the main CKD model below runs on time-series payer data. Hemoglobin and PCV are the strongest non-renal
signals, which fits the anemia that comes with CKD.

## Results (seed 20261029; 50,000 synthetic members; test set uses held-out members *and* later dates)

**Primary: 12-month recorded deterioration** (≥40% eGFR decline, eGFR<15, or dialysis). Test prevalence is 5.9%.

| Model | AUROC (95% CI) | PR-AUC | Brier | Top-10% capture |
|---|---|---|---|---|
| **PredictMed (LightGBM + twin)** | **0.897 (0.876–0.915)** | **0.589** | **0.0348** | **67.5%** |
| KFRE 4-variable (recalibrated) | 0.868 | 0.392 | 0.0424 | 60.7% |
| Logistic regression (same features) | 0.896 | 0.551 | 0.0368 | 67.2% |
| Prevalence | 0.500 | 0.059 | 0.0554 | 10.3% |

- **Calibration:** slope 1.01, ECE 0.006. In the eGFR<60 subgroup, AUROC is 0.886 vs 0.831 for KFRE.
- **Ascertainment gap:** only **76%** of true deteriorations were *recorded* in payer data. Against the true label, the model's AUROC is 0.967.
- **Conformal (α = 0.10) on test:** coverage is 91.9% for events and 89.3% for non-events.
- **Crash dialysis (24 months):** AUROC 0.96 on 36 events. This is exploratory.

**In-silico outreach trial** (as of 2023-07-01; 7,200 members not used in training; 60% uptake; 24-month horizon):

| Capacity | Risk-only targeting: deteriorations / dialysis / crash starts averted | **EAC targeting** |
|---|---|---|
| 250 | 9 / 10 / 3 | **21 / 19 / 8** |
| 500 | 27 / 22 / 7 | **47 / 27 / 12** |

Risk-only targeting picks members with *more* events. EAC targeting picks members whose events can actually be
prevented, and it averts about **2×** as many deteriorations.

**Digital twin compared with simulator ground truth:**
- Slope MAE is 2.16 mL/min/yr for the twin vs 7.13 for raw OLS.
- AUROC for detecting rapid progression is 0.79 vs 0.67.
- Level MAE is 5.05 for the twin vs 5.58 for the last lab value.

**FeatureScout:**
- It **caught the planted leakage canary** (`nephro_visits_90d`, whose ±90-day window reads the future). The feature's metadata claimed it was safe; the agent proved otherwise by re-running it on data truncated at the index date.
- It selected 28 of 58 features. The twin group alone added +0.14 PR-AUC over the claims baseline, which made the raw lab-trend group redundant.

**Streaming:**
- 28,319 events replayed with injected faults: 846 duplicates and 75 malformed events.
- 75 events went to the DLQ (classified by reason), 1,346 duplicates were suppressed, and 4 transient retries succeeded.
- Results are idempotent across a full redelivery.
- **Offline and online predictions agree exactly** (max |Δ| = 0).

**Real-data check, UCI Diabetes 130 (30-day readmission):**
- CatBoost test AUROC is **0.684** (0.669–0.696), and PR-AUC is 0.237 at a prevalence of 0.114.
- Calibration slope is 1.00.
- In the diabetic kidney disease subgroup, AUROC is 0.612.
- This is the honest benchmark: readmission from historical encounter data is hard to predict, and published work reports about 0.65–0.70.

## Layout

```
configs/config.yaml          single source of truth (seeds, splits, gates, capacity, costs)
src/predictmed/
  clinical.py                CKD-EPI 2021, KDIGO staging/heat map, KFRE, care-gap rules, trial priors
  simulator.py               literature-calibrated payer cohort + counterfactual intervention hook
  cohort.py                  eligibility, targets, oracle labels, hash-based member split
  features/                  point-in-time snapshots, kidney twin, dataset assembly
  lakehouse/iceberg_store.py Apache Iceberg (bronze/silver/gold, REST catalog, compaction, time travel)
  streaming/                 envelope + validation, Kafka/in-memory bus, state-writer/scorer/audit-sink
  agents/                    core runtime + HIL queue, DataSentinel, FeatureScout, ModelSmith, Guardian, CareNavigator
  models/                    candidate zoo (LightGBM, CatBoost, RF, PyTorch MLP), calibration/conformal/Venn-Abers, metrics
  ingestion/                 longitudinal adapters: Synthea (eGFR recomputed), MIMIC-IV v3.1, CMS DE-SynPUF
  repository/                PostgreSQL & SQLite repositories for state and HIL approvals
  auth/                      JWT authentication with roles (viewer, care_manager, model_approver)
  monitoring/                data drift and Population Stability Index (PSI) monitoring
  serving/api.py             FastAPI (result contract, quality status, HIL endpoints, serves React app)
frontend/                    Modern React + TypeScript + Vite + Tailwind CSS clinical dashboard
dags/predictmed_dag.py       Airflow DAG      slurm/        sbatch + job-array sweep
tests/                       35+ tests: adapters, fairness, modeling, leakage/canary, schemas, parity, E2E
docs/                        PLAN.md, MODEL_CARD.md, FAIRNESS.md, INFRASTRUCTURE.md
```

## Run

```bash
pip install -r requirements.txt
# UCI data (CC BY 4.0): put diabetic_data.csv in data/ (https://archive.ics.uci.edu/dataset/296)
python -m predictmed.pipeline run   # runs end-to-end pipeline tasks
pytest                              # full test suite (all unit & contract tests)
uvicorn predictmed.serving.api:app --port 8000   # serves API and React frontend on http://localhost:8000
```

Optional infrastructure:
- `docker compose up -d kafka mlflow`, then set `streaming.backend: kafka`.
- Mount `dags/` in Airflow.
- Set `PREDICTMED_USE_SLURM=1` to submit training through `sbatch`.
- Set `ANTHROPIC_API_KEY` (and optionally `PREDICTMED_LLM_MODEL`) so Claude writes the agent summaries and outreach briefs. Without a key, deterministic templates are used.

## Honest limitations

- The CKD results come from a simulator. They show that the methods work and how they compare, not how well the system would perform on real members.
- The outcome is *recorded* deterioration, and members who die or disenroll before the window ends are excluded.
- EAC uses literature priors. The in-silico trial checks the policy against the simulator's own mechanisms, which are calibrated to, but not identical with, those priors. Real validation needs a prospective (ideally randomized) outreach study.
- Costs in `config.yaml` are illustrative.
- Fairness: race is used only for auditing. Medicaid members show lower AUROC (0.84) and under-prediction (O/E 1.26), which is flagged in `reports/subgroups_test.csv`.
