"""Airflow DAG: batch ingestion -> Iceberg -> features -> agentic selection -> training (optionally on Slurm)
-> governance -> care navigation -> streaming parity check.

The task callables are the same functions the local runner uses (predictmed.pipeline.TASKS), so the
orchestrated and local paths cannot drift.  Set PREDICTMED_USE_SLURM=1 to submit `train` via sbatch
(slurm/train_array.sbatch) and wait on the job instead of running in the worker.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

ROOT = os.environ.get("PREDICTMED_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _run(task_name: str, **_):
    from predictmed.pipeline import TASKS, Ctx
    TASKS[task_name](Ctx())


def _train(**_):
    if os.environ.get("PREDICTMED_USE_SLURM") == "1":
        out = subprocess.run(["sbatch", "--parsable", "--wait", os.path.join(ROOT, "slurm", "train.sbatch")],
                             check=True, capture_output=True, text=True)
        print("slurm job", out.stdout.strip())
    else:
        _run("train")


default_args = {"owner": "predictmed", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG("predictmed_ckd", start_date=datetime(2026, 1, 1), schedule="@monthly", catchup=False,
         default_args=default_args, max_active_runs=1, tags=["ckd", "payer", "agentic"]) as dag:
    ingest = PythonOperator(task_id="ingest_to_iceberg", python_callable=_run, op_kwargs={"task_name": "ingest"})
    features = PythonOperator(task_id="point_in_time_features", python_callable=_run, op_kwargs={"task_name": "features"})
    select = PythonOperator(task_id="feature_scout_agent", python_callable=_run, op_kwargs={"task_name": "select"})
    train = PythonOperator(task_id="model_smith_and_guardian", python_callable=_train)
    explain = PythonOperator(task_id="explain_and_twin_validation", python_callable=_run, op_kwargs={"task_name": "explain"})
    care = PythonOperator(task_id="care_navigator_worklist", python_callable=_run, op_kwargs={"task_name": "care"})
    trial = PythonOperator(task_id="insilico_outreach_trial", python_callable=_run, op_kwargs={"task_name": "trial"})
    stream = PythonOperator(task_id="streaming_parity_check", python_callable=_run, op_kwargs={"task_name": "stream"})
    uci = PythonOperator(task_id="uci_real_data_track", python_callable=_run, op_kwargs={"task_name": "uci"})

    ingest >> features >> select >> train >> [explain, care, stream]
    care >> trial
    uci
