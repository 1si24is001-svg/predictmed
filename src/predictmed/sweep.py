"""One cell of the Slurm sweep: grouped-CV a (model, seed) on the training split, log to MLflow."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .models.candidates import make_candidate
from .models.tracking import Tracker
from .pipeline import Ctx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ctx = Ctx()
    frame = pd.read_parquet(ctx.D("frame.parquet"))
    feats = json.load(open(ctx.R("feature_manifest.json")))["features"]
    target = "y_deterioration_12m"
    tr = frame[(frame.split == "train") & frame[target].notna()].reset_index(drop=True)
    pr, roc = [], []
    for i, (a_, b_) in enumerate(GroupKFold(5).split(tr, tr[target], tr.member_id)):
        m = make_candidate(a.model, seed=a.seed).fit(tr.loc[a_, feats], tr.loc[a_, target])
        p = m.predict_proba(tr.loc[b_, feats])[:, 1]
        pr.append(average_precision_score(tr.loc[b_, target], p))
        roc.append(roc_auc_score(tr.loc[b_, target], p))
    t = Tracker(ctx.p["mlflow_uri"], "predictmed-sweep")
    with t.run(f"{a.model}-seed{a.seed}", params={"model": a.model, "seed": a.seed}, tags={"stage": "sweep"}):
        t.metrics({"cv_auprc": np.mean(pr), "cv_auprc_sd": np.std(pr), "cv_auroc": np.mean(roc)})
    print(json.dumps({"model": a.model, "seed": a.seed, "cv_auprc": float(np.mean(pr)), "cv_auroc": float(np.mean(roc))}))


if __name__ == "__main__":
    main()
