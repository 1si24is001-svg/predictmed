"""In-silico randomized outreach trial (simulation privilege, impossible with retrospective payer data).

Question: at equal care-manager capacity, does impact-weighted targeting (EAC) avert more *true* kidney
deterioration than risk-only targeting?

Method: the simulator uses common random numbers, so re-running it with an intervention changes outcomes
*only* through the intervention's causal pathway.  For each strategy we pick the top-K members at the as-of date,
apply their open care gaps with a fixed (hash-based) uptake rate, re-simulate, and compare true outcomes against
the factual world member by member.  This is a test of the *policy*, using the simulator's own mechanisms (which
are calibrated to, but not identical with, the literature priors that EAC uses)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .simulator import simulate


def _uptake(member_id: str, rate: float, seed: int) -> bool:
    return int(hashlib.sha256(f"uptake|{seed}|{member_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < rate


def true_outcomes(truth: pd.DataFrame, life: pd.DataFrame, members: list, T: pd.Timestamp, months: int) -> pd.DataFrame:
    tr = truth[truth.member_id.isin(members)].set_index(["member_id", "month"]).sort_index()
    t0, t1 = T - pd.DateOffset(months=1), T + pd.DateOffset(months=months - 1)
    out = []
    dial = life[(life.event == "DIALYSIS_START") & life.member_id.isin(members)].set_index("member_id")
    for mid in members:
        try:
            a = tr.loc[(mid, t0)].egfr_true
        except KeyError:
            out.append((mid, np.nan, np.nan, np.nan))
            continue
        g = tr.loc[mid]
        w = g[(g.index >= T) & (g.index <= t1)]
        det = bool(((w.egfr_true <= 0.6 * a) | (w.egfr_true < 15) | w.on_dialysis).any())
        d = dial.loc[[mid]] if mid in dial.index else None
        in_win = d is not None and ((d.event_month >= T) & (d.event_month <= t1)).any()
        crash = d is not None and (((d.event_month >= T) & (d.event_month <= t1)) & d.crash).any()
        out.append((mid, int(det), int(in_win), int(crash)))
    return pd.DataFrame(out, columns=["member_id", "deteriorated", "dialysis", "crash_dialysis"]).set_index("member_id")


def run_trial(cfg: dict, scored: pd.DataFrame, T: pd.Timestamp, capacities=(250, 500), uptake=0.6,
              horizon=24, log=print) -> dict:
    s = cfg["simulation"]
    m_T = (T.year - pd.Timestamp(s["start"]).year) * 12 + (T.month - pd.Timestamp(s["start"]).month)
    base = simulate(s["n_members"], s["months"], s["start"], s["rapid_progressor_share"], seed=cfg["seed"])
    res = {"as_of": str(T.date()), "uptake": uptake, "horizon_months": horizon, "population": len(scored),
           "strategies": {}}
    for K in capacities:
        for strat, key in (("risk_only", "p_deterioration"), ("eac", "eac_usd")):
            top = scored.sort_values(key, ascending=False).head(K)
            actions = {}
            for r in top.itertuples(index=False):
                gaps = [g for g in str(r.care_gaps).split("|") if g]
                if gaps and _uptake(r.member_id, uptake, cfg["seed"]):
                    actions[int(r.member_id[1:]) - 100000] = gaps
            cf = simulate(s["n_members"], s["months"], s["start"], s["rapid_progressor_share"], seed=cfg["seed"],
                          intervention={"month": m_T, "actions": actions})
            mem = top.member_id.tolist()
            f = true_outcomes(base["oracle_truth"], base["life_events"], mem, T, horizon)
            c = true_outcomes(cf["oracle_truth"], cf["life_events"], mem, T, horizon)
            averted = (f - c).sum()
            res["strategies"][f"{strat}@{K}"] = {
                "K": K, "members_intervened": len(actions), "factual_deteriorations": int(f.deteriorated.sum()),
                "deteriorations_averted": int(averted.deteriorated), "dialysis_starts_averted": int(averted.dialysis),
                "crash_starts_averted": int(averted.crash_dialysis),
                "selected_with_actionable_gap": int((top.modifiable_fraction > 0).sum()),
                "mean_p_deterioration": float(top.p_deterioration.mean())}
            log(f"  trial {strat:9s} K={K}: intervened {len(actions)}, factual events {int(f.deteriorated.sum())}, "
                f"averted deteriorations {int(averted.deteriorated)}, dialysis {int(averted.dialysis)}, "
                f"crash {int(averted.crash_dialysis)}")
    return res
