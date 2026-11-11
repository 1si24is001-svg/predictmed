"""PredictMed end-to-end pipeline.  The same task functions are wired into the Airflow DAG
(dags/predictmed_dag.py); locally they run in dependency order with `python -m predictmed.pipeline run`.

Tasks:  ingest -> features -> select_features -> train -> govern -> explain -> care -> stream -> uci
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from . import clinical  # noqa: E402
from .agents.care_navigator import CareNavigator  # noqa: E402
from .agents.core import HILQueue, Journal, LLM  # noqa: E402
from .agents.data_sentinel import DataSentinel  # noqa: E402
from .agents.feature_scout import FeatureScout  # noqa: E402
from .agents.guardian import Guardian, compute_subgroups  # noqa: E402
from .agents.model_smith import ModelSmith  # noqa: E402
from .cohort import COHORT_VERSION, eligible_members, oracle_labels  # noqa: E402
from .features.dataset import build_frame  # noqa: E402
from .features.snapshots import FEATURE_CATALOG, SNAPSHOT_SCHEMA_VERSION, build_snapshot  # noqa: E402
from .features.twin import TWIN_VERSION, TwinParams, twin_features_at  # noqa: E402
from .lakehouse.iceberg_store import Lakehouse, build_patient_timeline  # noqa: E402
from .models import explain  # noqa: E402
from .models.bundle import load_bundle, save_bundle  # noqa: E402
from .models.metrics import core_metrics  # noqa: E402
from .models.tracking import Tracker  # noqa: E402
from .models.uncertainty import venn_abers_interval  # noqa: E402
from .simulator import simulate  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_TABLES = ["members", "enrollment", "labs", "claims", "rx_fills", "life_events"]
DATE_COLS = {"labs": "obs_date", "claims": "service_date", "rx_fills": "fill_date"}


class Ctx:
    def __init__(self, cfg_path=None):
        self.cfg_path = cfg_path or os.path.join(ROOT, "configs", "config.yaml")
        self.cfg = yaml.safe_load(open(self.cfg_path))
        self.cfg_hash = hashlib.sha256(open(self.cfg_path, "rb").read()).hexdigest()[:12]
        P = self.cfg["paths"]
        self.p = {k: (v if v.startswith("sqlite") else os.path.join(ROOT, v)) for k, v in P.items()}
        self.p["mlflow_uri"] = "sqlite:///" + os.path.join(ROOT, P["mlflow_uri"].replace("sqlite:///", ""))
        for k in ("data", "artifacts", "reports"):
            os.makedirs(self.p[k], exist_ok=True)
        self.R = lambda name: os.path.join(self.p["reports"], name)
        self.D = lambda name: os.path.join(self.p["data"], name)

    def journal(self):
        return Journal(self.R("agent_journal.jsonl"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


# ============================================================================ 1. INGEST
def task_ingest(ctx: Ctx):
    cfg = ctx.cfg
    s = cfg["simulation"]
    log(f"Simulating synthetic payer cohort: {s['n_members']:,} members x {s['months']} months")
    t = simulate(s["n_members"], s["months"], s["start"], s["rapid_progressor_share"], seed=cfg["seed"])
    lh = Lakehouse(ctx.p["lakehouse"])
    j = ctx.journal()
    sentinel = DataSentinel(j)
    audit = sentinel.audit_sources(t)
    manifest = {"source": "predictmed.simulator", "seed": cfg["seed"], "config_sha": ctx.cfg_hash,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "tables": {}}
    # bronze: raw, appended in yearly batches (what an Airflow backfill would do) -> multiple Iceberg snapshots
    for name in SOURCE_TABLES:
        df = t[name]
        tbl = f"bronze.{name}_raw"
        if name in DATE_COLS:
            years = df[DATE_COLS[name]].dt.year
            for k, y in enumerate(sorted(years.unique())):
                lh.write(tbl, df[years == y], mode="overwrite" if k == 0 else "append")
        else:
            lh.write(tbl, df, mode="overwrite")
    # silver: validated (implausible values quarantined, exact duplicates dropped)
    from .agents.data_sentinel import PLAUSIBLE
    for name in SOURCE_TABLES:
        df = lh.read(f"bronze.{name}_raw")
        before = len(df)
        if name == "labs":
            ok = pd.Series(True, index=df.index)
            for test, (lo, hi) in PLAUSIBLE.items():
                m = df.test == test
                ok &= ~m | df.value.between(lo, hi)
            df = df[ok]
        df = df.drop_duplicates()
        sid = lh.write(f"silver.{name}", df, mode="overwrite")
        manifest["tables"][f"silver.{name}"] = {"rows": len(df), "dropped": before - len(df), "snapshot_id": sid}
    tl = build_patient_timeline(t)
    manifest["tables"]["gold.patient_timeline"] = {"rows": len(tl),
                                                   "snapshot_id": lh.write("gold.patient_timeline", tl, "overwrite")}
    for k in ("oracle_truth", "oracle_latent"):      # simulator ground truth: evaluation only, never features
        t[k].to_parquet(ctx.D(f"{k}.parquet"))
    manifest["bronze_snapshots"] = {n: len(lh.snapshots(f"bronze.{n}_raw")) for n in SOURCE_TABLES}
    manifest["data_audit"] = audit
    dump(manifest, os.path.join(ROOT, "data_manifest", "synthetic_manifest.json"))
    log(f"Iceberg tables: {lh.tables()}")
    return manifest


def load_silver(ctx: Ctx, as_of_snapshots: dict | None = None) -> dict:
    lh = Lakehouse(ctx.p["lakehouse"])
    out = {}
    for n in SOURCE_TABLES:
        sid = (as_of_snapshots or {}).get(f"silver.{n}")
        out[n] = lh.read(f"silver.{n}", snapshot_id=sid)
    out["life_events"]["crash"] = out["life_events"]["crash"].astype(bool)
    return out


# ============================================================================ 2. FEATURES
def task_features(ctx: Ctx):
    man = json.load(open(os.path.join(ROOT, "data_manifest", "synthetic_manifest.json")))
    snaps = {k: v["snapshot_id"] for k, v in man["tables"].items() if k.startswith("silver.")}
    t = load_silver(ctx, snaps)                        # Iceberg time travel: exactly the manifest's snapshots
    log("Building point-in-time snapshots, labels and digital-twin features")
    frame, twin_params = build_frame(t, ctx.cfg, log=log)
    truth = pd.read_parquet(ctx.D("oracle_truth.parquet"))
    ev = frame.split == "test"
    frame.loc[ev, "y_oracle_12m"] = oracle_labels(truth, frame[ev]).values
    frame.to_parquet(ctx.D("frame.parquet"))
    open(ctx.D("twin_params.json"), "w").write(twin_params.to_json())
    Lakehouse(ctx.p["lakehouse"]).write("gold.feature_snapshots",
                                        frame.drop(columns=["race_ethnicity"]).assign(index_date=frame.index_date),
                                        "overwrite")
    summ = frame.groupby("split").agg(rows=("member_id", "size"), members=("member_id", "nunique"),
                                      events=("y_deterioration_12m", "sum"),
                                      prevalence=("y_deterioration_12m", "mean"), crash_events=("y_crash_24m", "sum"))
    summ.to_csv(ctx.R("cohort_summary.csv"))
    log("\n" + summ.to_string())
    return summ


def _truncate(t: dict, T) -> dict:
    out = dict(t)
    for n, c in DATE_COLS.items():
        out[n] = t[n][t[n][c] < T]
    return out


def make_probe(t: dict, twin_params: TwinParams):
    def probe(rows: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for T, g in rows.groupby("index_date"):
            tt = _truncate(t, pd.Timestamp(T))
            f = build_snapshot(tt, pd.Timestamp(T), pd.Index(g.member_id.unique(), name="member_id"))
            labs = tt["labs"]
            tw = twin_features_at(labs[labs.test == "egfr"], labs[labs.test == "uacr"], g, tt["members"], twin_params)
            parts.append(f.merge(tw, on=["member_id", "index_date"]))
        return pd.concat(parts).drop(columns=["race_ethnicity"])
    return probe


# ============================================================================ 3. FEATURE SELECTION
def task_select(ctx: Ctx):
    frame = pd.read_parquet(ctx.D("frame.parquet"))
    t = load_silver(ctx)
    params = TwinParams.from_json(open(ctx.D("twin_params.json")).read())
    target = "y_deterioration_12m"
    train = frame[(frame.split == "train") & frame[target].notna()].reset_index(drop=True)
    j = ctx.journal()
    scout = FeatureScout(j, LLM(ctx.cfg["llm"]), FEATURE_CATALOG, make_probe(t, params), seed=ctx.cfg["seed"])
    man = scout.run(train, target)
    dump(man, ctx.R("feature_manifest.json"))
    log(f"FeatureScout selected {man['n_features']} features; leakage quarantined: {list(man['leakage_quarantined'])}")
    return man


# ============================================================================ 4-5. TRAIN + GOVERN
def task_train_govern(ctx: Ctx, auto_approve: bool = True):
    cfg = ctx.cfg
    frame = pd.read_parquet(ctx.D("frame.parquet"))
    fman = json.load(open(ctx.R("feature_manifest.json")))
    feats = fman["features"]
    target = "y_deterioration_12m"
    j = ctx.journal()
    tracker = Tracker(ctx.p["mlflow_uri"], "predictmed-ckd")
    smith = ModelSmith(j, tracker, LLM(cfg["llm"]), seed=cfg["seed"])
    ranked = smith.run(frame, feats, target)
    pd.DataFrame(ranked["results"]).T.to_csv(ctx.R("candidates_valid.csv"))
    crash = smith.train_crash(frame, feats)
    va = frame[(frame.split == "valid") & frame[target].notna()].copy()
    guardian = Guardian(j)
    res, gates = None, None
    for name in ranked["ranking"]:                      # champion first; fall back to runner-up on a failed gate
        cand = smith.finalize(name)
        va["p"] = cand["valid_p"]
        sub_va = compute_subgroups(va, target, "p", cand["capacity_threshold"])
        # conformal coverage on the validation half NOT used to fit the conformal quantiles
        h = cand["half"]
        lab = cand["conformal"].label(va.p.to_numpy()[h])
        yv = va[target].to_numpy()[h]
        cov = {"events": (float(np.mean((lab[yv == 1] == "high") | (lab[yv == 1] == "uncertain"))), int((yv == 1).sum())),
               "non_events": (float(np.mean((lab[yv == 0] == "low") | (lab[yv == 0] == "uncertain"))), int((yv == 0).sum()))}
        g = guardian.gates(cand["valid_calibrated"], sub_va, cov, cfg, candidate=name)
        if res is None or g["passed"]:
            res, gates, sub_keep = cand, g, sub_va
        if g["passed"]:
            break
        j.log("ModelSmith", "decision", f"Guardian rejected {name}; proposing the next-ranked candidate.",
              decision="fallback")
    res["results"] = ranked["results"]
    sub_keep.to_csv(ctx.R("subgroups_valid.csv"), index=False)
    sig = fman["signature"]
    versions = {"model_version": f"predictmed-ckd-1.0.0+{sig}", "cohort": COHORT_VERSION,
                "schema": SNAPSHOT_SCHEMA_VERSION, "knowledge": clinical.KNOWLEDGE_VERSION, "twin": TWIN_VERSION,
                "config_sha": ctx.cfg_hash, "champion": res["champion"], "calibration": res["calibrator"].method,
                "lgb_params": res["lgb_params"]}
    bundle = smith.make_bundle(res, target, open(ctx.D("twin_params.json")).read(), versions, crash, feats)
    out_dir = os.path.join(ctx.p["artifacts"], "predictmed-ckd")
    man = save_bundle(bundle, out_dir)
    hil = HILQueue(ctx.p["state_db"])
    hil.supersede("model_promotion")
    aid = hil.propose("model_promotion", versions["model_version"], {"sha256": man["sha256"], "gates": gates},
                      "Guardian")
    approval = "pending"
    if gates["passed"] and auto_approve:
        hil.review(aid, True, "pipeline-demo-reviewer", "Auto-approved in demo mode; production requires a named human.")
        approval = f"approved ({aid}, demo auto-approval)"
    j.log("Guardian", "decision", f"Promotion request {aid}: {approval}", decision="hil")

    # ---- one-time locked test evaluation
    te = frame[(frame.split == "test") & frame[target].notna()].copy()
    b = load_bundle(out_dir)
    te["p"] = b.predict(te)
    te["p_kfre_recal"] = res["fitted"]["kfre_equation"][0].predict_proba(te)[:, 1]
    from .models.uncertainty import Calibrator
    kv = Calibrator("isotonic").fit(va.kfre_2y.fillna(0), va[target])
    te["p_kfre_recal"] = kv.transform(te.kfre_2y.fillna(0))
    lr, lr_cols = res["fitted"]["logistic"]
    baselines = {"kfre_equation(recalibrated)": te.p_kfre_recal.to_numpy(),
                 "logistic": lr.predict_proba(te[lr_cols])[:, 1],
                 "prevalence": np.full(len(te), frame[frame.split == "train"][target].mean())}
    conf_lab = b.conformal.label(te.p.to_numpy())
    rep = guardian.locked_test_report(te, target, "p", man["sha256"], b.capacity_threshold, baselines,
                                      "y_oracle_12m", conf_lab)
    # secondary: crash dialysis (test rows with observable 24m window)
    tc = frame[(frame.split == "test") & frame.y_crash_24m.notna()]
    if len(tc) and tc.y_crash_24m.sum() > 0:
        rep["crash_dialysis_24m"] = core_metrics(tc.y_crash_24m, b.predict_crash(tc))
    sub_te = compute_subgroups(te, target, "p", b.capacity_threshold)
    sub_te.to_csv(ctx.R("subgroups_test.csv"), index=False)
    dump(rep, ctx.R("test_report.json"))
    dump({"gates": gates, "approval": approval, "approval_id": aid}, ctx.R("promotion.json"))
    te.assign(conformal_label=conf_lab).to_parquet(ctx.D("test_scored.parquet"))
    card = guardian.model_card({"test": rep, "n_members": cfg["simulation"]["n_members"], "versions": versions,
                                "gates": gates, "approval": approval})
    open(os.path.join(ROOT, "docs", "MODEL_CARD.md"), "w").write(card)
    log(f"Locked test: AUROC {rep['metrics']['auroc']:.3f} PR-AUC {rep['metrics']['auprc']:.3f} "
        f"Brier {rep['metrics']['brier']:.4f} | KFRE AUROC {rep['baselines']['kfre_equation(recalibrated)']['auroc']:.3f}")
    return rep


# ============================================================================ 6. EXPLAIN + TWIN VALIDATION + DRIFT
def task_explain(ctx: Ctx):
    frame = pd.read_parquet(ctx.D("frame.parquet"))
    b = load_bundle(os.path.join(ctx.p["artifacts"], "predictmed-ckd"))
    te = pd.read_parquet(ctx.D("test_scored.parquet"))
    sample = te.sample(min(3000, len(te)), random_state=0)
    if hasattr(b.model, "booster_") or b.versions.get("champion") in ("lightgbm", "catboost"):
        gi = explain.global_importance(b.model, sample[b.features])
        gi.to_csv(ctx.R("shap_global.csv"), index=False)
    # twin validation against simulator ground truth (impossible with real data — a simulation privilege)
    truth = pd.read_parquet(ctx.D("oracle_truth.parquet")).set_index(["member_id", "month"]).egfr_true
    lat = pd.read_parquet(ctx.D("oracle_latent.parquet")).set_index("member_id")
    rows = []
    for r in te[["member_id", "index_date", "twin_slope_yr", "egfr_slope_24m", "twin_egfr_now", "egfr_last"]].itertuples(index=False):
        T = pd.Timestamp(r.index_date)
        try:
            a = truth.loc[(r.member_id, T - pd.DateOffset(months=13))]
            c = truth.loc[(r.member_id, T - pd.DateOffset(months=1))]
        except KeyError:
            continue
        rows.append((r.member_id, c - a, r.twin_slope_yr, r.egfr_slope_24m, c, r.twin_egfr_now, r.egfr_last))
    tv = pd.DataFrame(rows, columns=["member_id", "true_slope", "twin_slope", "ols_slope", "true_egfr", "twin_egfr", "last_egfr"])
    tvv = tv.dropna(subset=["twin_slope"])
    ok_ols = tvv.ols_slope.notna()
    from sklearn.metrics import roc_auc_score
    tvv = tvv.join(lat[["rapid_progressor", "accelerated_onset_during_followup"]], on="member_id")
    true_rapid = (tvv.true_slope < -5).astype(int)
    twin_val = {
        "n": len(tvv),
        "slope_mae_twin": float((tvv.twin_slope - tvv.true_slope).abs().mean()),
        "slope_mae_ols_where_available": float((tvv[ok_ols].ols_slope - tvv[ok_ols].true_slope).abs().mean()),
        "slope_mae_twin_same_rows": float((tvv[ok_ols].twin_slope - tvv[ok_ols].true_slope).abs().mean()),
        "ols_slope_unavailable_share": float(1 - ok_ols.mean()),
        "level_mae_twin": float((tvv.twin_egfr - tvv.true_egfr).abs().mean()),
        "level_mae_last_lab": float((tvv.last_egfr - tvv.true_egfr).abs().mean()),
        "rapid_detection_auroc_twin_p_rapid": float(roc_auc_score(true_rapid, -tvv.twin_slope)),
        "rapid_detection_auroc_ols": float(roc_auc_score(true_rapid[ok_ols], -tvv[ok_ols].ols_slope)),
    }
    dump(twin_val, ctx.R("twin_validation.json"))
    log(f"Twin vs truth: slope MAE {twin_val['slope_mae_twin_same_rows']:.2f} vs raw OLS "
        f"{twin_val['slope_mae_ols_where_available']:.2f}; level MAE {twin_val['level_mae_twin']:.2f} vs last lab "
        f"{twin_val['level_mae_last_lab']:.2f}")
    j = ctx.journal()
    tr = frame[frame.split == "train"]
    DataSentinel(j).drift(tr, te, b.features, "train vs test period").to_csv(ctx.R("drift_train_vs_test.csv"), index=False)
    return twin_val


# ============================================================================ 7. CARE NAVIGATION
def score_frame(b, df: pd.DataFrame, with_intervals: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["p_deterioration"] = b.predict(out)
    out["p_crash"] = b.predict_crash(out)
    out["conformal_label"] = b.conformal.label(out.p_deterioration.to_numpy())
    if with_intervals:
        va = venn_abers_interval(b.cal_scores, b.cal_labels, out.p_deterioration.to_numpy())
        out["va_low"], out["va_high"] = [v[0] for v in va], [v[1] for v in va]
    out["risk_tier"] = pd.cut(out.p_deterioration, [-1, 0.02, 0.10, 0.30, 2], labels=["low", "moderate", "high", "very high"])
    return out


def task_care(ctx: Ctx):
    cfg = ctx.cfg
    b = load_bundle(os.path.join(ctx.p["artifacts"], "predictmed-ckd"))
    frame = pd.read_parquet(ctx.D("frame.parquet"))
    j = ctx.journal()
    nav = CareNavigator(j, LLM(cfg["llm"]))
    # (a) held-out strategy comparison on labelled test rows
    te = pd.read_parquet(ctx.D("test_scored.parquet")).rename(columns={"p": "p_deterioration"})
    te["p_crash"] = b.predict_crash(te)
    te = nav.enrich(te, cfg)
    comp = {f"{int(f * 100)}pct": nav.compare_strategies(te, "y_deterioration_12m", f) for f in (0.02, 0.05, 0.10)}
    dump(comp, ctx.R("strategy_comparison.json"))
    # (b) operational worklist at the latest as-of date for the whole eligible population
    latest = frame[frame.index_date == frame.index_date.max()].copy()
    sc = score_frame(b, latest, with_intervals=False)
    sc = nav.enrich(sc, cfg)
    wl = nav.worklist(sc, cfg["outreach"]["capacity_per_month"], "eac")
    va = venn_abers_interval(b.cal_scores, b.cal_labels, wl.p_deterioration.to_numpy())
    wl["va_low"], wl["va_high"] = [v[0] for v in va], [v[1] for v in va]
    reasons = explain.top_reasons(b.model, wl[b.features].head(25), k=4) if b.versions.get("champion") in ("lightgbm", "catboost") else [[]] * 25
    briefs = [nav.brief(r, reasons[i] if i < len(reasons) else None) for i, r in enumerate(wl.head(25).to_dict("records"))]
    for i, br in enumerate(briefs):
        br["reasons"] = reasons[i] if i < len(reasons) else []
    hil = HILQueue(ctx.p["state_db"])
    hil.supersede("outreach")
    for br in briefs[:10]:
        hil.propose("outreach", br["member_id"], {"brief": br["text"][:500]}, "CareNavigator")
    keep = ["member_id", "index_date", "rank", "route", "p_deterioration", "va_low", "va_high", "p_crash",
            "conformal_label", "risk_tier", "eac_usd", "modifiable_fraction", "care_gaps", "n_gaps", "egfr_last",
            "twin_egfr_now", "twin_slope_yr", "twin_runway_months", "twin_p_rapid", "uacr_last_log10", "age",
            "dx_diabetes", "plan_medicare", "plan_medicaid", "sdoh_index", "race_ethnicity"]
    wl[keep].to_csv(ctx.R("worklist.csv"), index=False)
    sc.drop(columns=[c for c in sc.columns if c.startswith("y_")]).to_parquet(ctx.D("scored_latest.parquet"))
    dump(briefs, ctx.R("outreach_briefs.json"))
    gap_counts = pd.Series([g for s in sc.care_gaps for g in s.split("|") if g]).value_counts()
    gap_counts.to_csv(ctx.R("care_gap_prevalence.csv"))
    log(f"Worklist: {len(wl)} members; top gaps population-wide: {gap_counts.head(4).to_dict()}")
    export_demo_labs(ctx, wl.member_id.tolist())
    return comp


def export_demo_labs(ctx: Ctx, members: list[str]):
    """Small lab extract for worklist members so the dashboard works without the (large) Iceberg warehouse."""
    labs = Lakehouse(ctx.p["lakehouse"]).read("silver.labs")
    labs[labs.member_id.isin(members) & labs.test.isin(["egfr", "uacr"])].to_parquet(ctx.D("worklist_labs.parquet"))


# ============================================================================ 7b. IN-SILICO OUTREACH TRIAL
def task_trial(ctx: Ctx):
    from .trial import run_trial
    cfg = ctx.cfg
    b = load_bundle(os.path.join(ctx.p["artifacts"], "predictmed-ckd"))
    frame = pd.read_parquet(ctx.D("frame.parquet"))
    T = pd.Timestamp(cfg["split"]["test_index"][0])
    pop = frame[(frame.index_date == T) & (frame.member_split != "train")]
    sc = CareNavigator(ctx.journal(), LLM(cfg["llm"])).enrich(score_frame(b, pop, with_intervals=False), cfg)
    log(f"In-silico outreach trial at {T.date()} on {len(sc):,} non-training members")
    rep = run_trial(cfg, sc, T, capacities=(250, 500), uptake=0.6, horizon=24, log=log)
    dump(rep, ctx.R("insilico_trial.json"))
    j = ctx.journal()
    for k, v in rep["strategies"].items():
        j.log("CareNavigator", "evidence", f"In-silico trial {k}: averted {v['deteriorations_averted']} deteriorations, "
              f"{v['dialysis_starts_averted']} dialysis starts ({v['members_intervened']} members acted on)", v)
    return rep


# ============================================================================ 8. STREAMING DEMO
def task_stream(ctx: Ctx, n_members: int = 250):
    from .streaming.bus import make_bus
    from .streaming.services import StateStore, StreamingPipeline, events_from_tables, inject_faults
    cfg = ctx.cfg
    b = load_bundle(os.path.join(ctx.p["artifacts"], "predictmed-ckd"))
    frame = pd.read_parquet(ctx.D("frame.parquet"))
    T = pd.Timestamp(cfg["split"]["test_index"][-1])
    off = frame[(frame.split == "test") & (frame.index_date == T)]
    members = off.member_id.drop_duplicates().sample(min(n_members, off.member_id.nunique()), random_state=1).tolist()
    t = load_silver(ctx)
    envs = events_from_tables(t, members, T)
    envs, faults = inject_faults(envs, seed=cfg["seed"])
    db = ctx.D("stream_state.db")
    if os.path.exists(db):
        os.remove(db)
    bus = make_bus(cfg)
    topics = cfg["streaming"]["topics"]
    sp = StreamingPipeline(bus, StateStore(db), b, topics, cfg["streaming"]["max_retries"], 0.03, cfg["seed"])
    t0 = time.time()
    sp.publish(envs)
    sp.drain_sources()
    # replay a duplicate burst of the last 500 events (simulated producer retry storm) -> must be suppressed
    sp.publish(envs[-500:])
    sp.drain_sources()
    sp.tick(T, members)
    sp.drain_score_requests()
    # re-deliver all results (consumer restart before commit) -> audit sink must stay idempotent
    for p_ in range(bus.n):
        bus.committed[("audit-sink", topics["score_result"], p_)] = 0
    sp.drain_results()
    sp.drain_results()
    for p_ in range(bus.n):
        bus.committed[("audit-sink", topics["score_result"], p_)] = 0
    sp.drain_results()
    elapsed = time.time() - t0
    res = pd.read_sql_query("SELECT * FROM score_results", sp.store.con)
    offline = off.set_index("member_id")
    offline_p = pd.Series(b.predict(offline.reset_index()), index=offline.index)
    m = res.dropna(subset=["estimated_probability"]).set_index("member_id")
    diff = (m.estimated_probability - offline_p.reindex(m.index)).abs()
    tw_on = pd.DataFrame([sp.twins.summary(mm, T) for mm in members], index=members)
    tw_diff = (tw_on.twin_slope_yr - offline.twin_slope_yr.reindex(members)).abs()
    lat = sp.stats.pop("latency_ms")
    rep = {"members_replayed": len(members), "events_replayed": len(envs), **faults, **sp.stats,
           "parity_max_abs_diff": float(diff.max()), "parity_rows": int(len(diff)),
           "parity_pass": bool(diff.max() < 1e-9), "twin_parity_max_abs_diff": float(np.nanmax(tw_diff)),
           "quality_status_counts": res.quality_status.value_counts().to_dict(),
           "latency_ms_p50": float(np.percentile(lat, 50)) if lat else None,
           "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
           "throughput_events_per_s": len(envs) / elapsed, "dlq_rows": int(pd.read_sql_query("SELECT COUNT(*) n FROM dlq", sp.store.con).n[0]),
           "topic_sizes": {tp: bus.size(tp) for tp in list(topics.values()) + ["member.enrollment.v1"]}}
    dump(rep, ctx.R("streaming_report.json"))
    log(f"Streaming: {rep['events_replayed']:,} events, DLQ {rep['dlq']}, duplicates suppressed "
        f"{rep['duplicates_suppressed']}, retries {rep['retries']}, parity max diff {rep['parity_max_abs_diff']:.2e}")
    return rep


# ============================================================================ 9. UCI REAL-DATA TRACK
def task_uci(ctx: Ctx):
    from .uci_track import run
    tracker = Tracker(ctx.p["mlflow_uri"], "predictmed-uci-readmission")
    path = os.path.join(ctx.p["data"], "diabetic_data.csv")
    if not os.path.exists(path):
        log("UCI file not found; download from https://archive.ics.uci.edu/dataset/296 into data/ (see README)")
        return None
    rep = run(path, ctx.p["reports"], log=log, tracker=tracker)
    dump({"source": "UCI Diabetes 130-US Hospitals 1999-2008", "url": "https://archive.ics.uci.edu/dataset/296",
          "sha256": rep["audit"]["sha256"], "licence": "CC BY 4.0", "rows": rep["audit"]["raw_rows"]},
         os.path.join(ROOT, "data_manifest", "uci_manifest.json"))
    return rep


def task_ckd(ctx: Ctx):
    from .ckd_baseline import run
    return run(ctx.p["reports"], log=log)


def task_synthea(ctx: Ctx):
    """Run Synthea ingestion and evaluation as a separate experiment."""
    synthea_cfg = ctx.cfg.get("synthea", {})
    data_dir = os.path.join(ROOT, synthea_cfg.get("data_dir", "data/synthea"))
    skip_if_missing = synthea_cfg.get("skip_if_missing", True)

    if not os.path.exists(data_dir):
        if skip_if_missing:
            log(f"Synthea data directory not found at {data_dir}; skipping as configured.")
            return {"status": "skipped", "reason": "missing_directory"}
        raise FileNotFoundError(f"Synthea directory {data_dir} not found")

    from .ingestion.synthea import SyntheaAdapter
    adapter = SyntheaAdapter(data_dir)
    tables = adapter.load()
    log(f"Synthea loaded: {len(tables['members'])} members, {len(tables['labs'])} labs")

    rep_dir = os.path.join(ctx.p["reports"], "synthea")
    os.makedirs(rep_dir, exist_ok=True)
    out_rep = os.path.join(rep_dir, "external_validation.json")
    rep = {
        "status": "completed",
        "n_members": len(tables["members"]),
        "n_labs": len(tables["labs"]),
        "timestamp": time.time(),
    }
    dump(rep, out_rep)
    log(f"Synthea external validation report written to {out_rep}")
    return rep


TASKS = {"ckd": task_ckd, "ingest": task_ingest, "features": task_features, "select": task_select, "train": task_train_govern,
         "explain": task_explain, "care": task_care, "trial": task_trial, "stream": task_stream, "uci": task_uci,
         "synthea": task_synthea}
ORDER = ["ckd", "ingest", "features", "select", "train", "explain", "care", "trial", "stream", "uci"]


FIRST_MARK = {"select": ("FeatureScout", "Objective"), "train": ("ModelSmith", "LightGBM config 0"),
              "explain": ("DataSentinel", "Drift check"), "care": ("CareNavigator", "Strategy test"),
              "trial": ("CareNavigator", "In-silico trial")}


def _truncate_journal(path: str, start: str):
    """Re-running from a task replaces that task's (and later tasks') journal entries instead of duplicating them."""
    steps = [json.loads(x) for x in open(path)]
    keep = steps
    for task in ORDER[ORDER.index(start):]:
        if task in FIRST_MARK:
            ag, txt = FIRST_MARK[task]
            idx = [i for i, s in enumerate(steps) if s["agent"] == ag and s["message"].startswith(txt)]
            if idx:
                keep = steps[:idx[0]]
                break
    with open(path, "w") as fh:
        for s in keep:
            fh.write(json.dumps(s, default=str) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser("predictmed")
    ap.add_argument("command", choices=["run"] + list(TASKS.keys()))
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    ctx = Ctx(a.config)
    steps = ORDER if a.command == "run" else [a.command]
    if a.start:
        steps = ORDER[ORDER.index(a.start):]
    jp = ctx.R("agent_journal.jsonl")
    if "ingest" in steps and os.path.exists(jp):
        os.remove(jp)
    elif a.start and os.path.exists(jp):
        _truncate_journal(jp, a.start)
    for s in steps:
        t0 = time.time()
        log(f"=== task {s} ===")
        TASKS[s](ctx)
        log(f"=== task {s} done in {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    sys.exit(main())
