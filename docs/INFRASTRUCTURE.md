# PredictMed Production Infrastructure Guide

## Architecture Overview

PredictMed supports dual operational modes:
1. **Embedded / Local Mode (Default)**: Embedded SQLite lakehouse catalog, in-memory partitioned bus, local state database, and FastAPI + Vite React frontend.
2. **Distributed Production Mode (`docker-compose.yml`)**:
   - **Kafka (KRaft)**: Event streaming with 6 partitions per topic, deterministic subject-key ordering, and DLQ enforcement.
   - **PostgreSQL 16**: High-concurrency relational store for HIL approvals, score results, and idempotency deduplication.
   - **MinIO & Iceberg REST Catalog**: S3-compatible object store backing Apache Iceberg REST catalog for petabyte-scale longitudinal timelines with data compaction and snapshot expiration.
   - **MLflow Tracking Server**: Centralized metric logging and model registry.
   - **FastAPI Backend & React Frontend**: Containerized web platform with role-based JWT auth.

## Slurm Job Execution

PredictMed includes Slurm batch definitions in `slurm/`:
- `slurm/train.sbatch`: Submits end-to-end model training, validation calibration, and Guardian promotion evaluation.
- `slurm/sweep_array.sbatch`: Slurm array job (20 tasks) sweeping 5 candidate architectures across random seeds with member-grouped cross-validation.

### Testing on Windows / Developer Machines
Slurm requires the Linux Slurm controller daemon (`slurmctld`), munge authentication, and slurmd compute workers (`sbatch`).
To run locally on a developer machine:
```bash
# Using giovtorres/slurm-docker-cluster on Docker with Linux containers / WSL2:
git clone https://github.com/giovtorres/slurm-docker-cluster
cd slurm-docker-cluster
docker compose up -d
docker exec -it slurmctld bash -c "cd /predictmed && sbatch slurm/train.sbatch"
```
On Windows without a Linux container engine, the pipeline tasks can be invoked directly and deterministically via:
```bash
python -m predictmed.pipeline train
python -m predictmed.sweep --model lightgbm --seed 0
```

