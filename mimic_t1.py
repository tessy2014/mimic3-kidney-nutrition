"""
Frequency and timing of dietitian assessment in critically ill patients with acute or
chronic kidney disease (MIMIC-III v1.4, MetaVision era).

Dietitian assessment is identified from the timestamps of notes in the 'Nutrition'
category of NOTEEVENTS. The TEXT column is never read. Only files named summary_*
(aggregate counts, 1-10 shown as "<11") may be shared.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import mimic_t3 as m3

SMALL = m3.SMALL
GROUPS = m3.GROUPS
CIF_HOURS = list(range(0, 169, 6))


# ------------------------------------------------------------------- note times
def nutrition_note_times(d, hadm_ids, chunksize=200_000):
    """Times of Nutrition notes. TEXT is excluded with usecols and never loaded."""
    cols = ["SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "STORETIME", "CATEGORY", "ISERROR"]
    keep = set(hadm_ids)
    parts = []
    for ch in pd.read_csv(Path(d) / "NOTEEVENTS.csv.gz", usecols=cols, chunksize=chunksize,
                          dtype={"CATEGORY": str}, low_memory=False):
        ch = ch[(ch.CATEGORY.str.strip() == "Nutrition") & (ch.ISERROR.fillna(0) != 1)
                & ch.HADM_ID.isin(keep)]
        parts.append(ch)
    n = pd.concat(parts, ignore_index=True)
    for c in ["CHARTDATE", "CHARTTIME", "STORETIME"]:
        n[c] = pd.to_datetime(n[c], errors="coerce")
    # Time of assessment: chart time; else store time; else end of the chart date
    n["NOTE_TIME"] = n.CHARTTIME.fillna(n.STORETIME).fillna(
        n.CHARTDATE + pd.Timedelta(hours=23, minutes=59))
    n["TIME_SOURCE"] = np.select([n.CHARTTIME.notna(), n.STORETIME.notna()],
                                 ["CHARTTIME", "STORETIME"], default="CHARTDATE only")
    return n[["HADM_ID", "NOTE_TIME", "TIME_SOURCE"]]


def assessment_by_stay(cohort, notes):
    """First assessment during the ICU stay, note counts, and pre-ICU assessment."""
    x = notes.merge(cohort[["ICUSTAY_ID", "HADM_ID", "INTIME", "OUTTIME"]], on="HADM_ID")
    in_icu = x[(x.NOTE_TIME >= x.INTIME) & (x.NOTE_TIME <= x.OUTTIME)]
    pre = x[x.NOTE_TIME < x.INTIME]
    first = in_icu.sort_values("NOTE_TIME").drop_duplicates("ICUSTAY_ID")
    out = cohort[["ICUSTAY_ID"]].copy()
    out = out.merge(first[["ICUSTAY_ID", "NOTE_TIME", "TIME_SOURCE"]]
                    .rename(columns={"NOTE_TIME": "FIRST_NOTE", "TIME_SOURCE": "FIRST_SOURCE"}),
                    on="ICUSTAY_ID", how="left")
    out = out.merge(in_icu.groupby("ICUSTAY_ID").size().rename("N_NOTES_ICU").reset_index(),
                    on="ICUSTAY_ID", how="left")
    out["N_NOTES_ICU"] = out.N_NOTES_ICU.fillna(0).astype(int)
    out["PRE_ICU_NOTE"] = out.ICUSTAY_ID.isin(pre.ICUSTAY_ID)
    return out


def add_times(c):
    c = c.copy()
    c["LOS_H"] = (c.OUTTIME - c.INTIME).dt.total_seconds() / 3600
    c["HOURS_TO_NOTE"] = (c.FIRST_NOTE - c.INTIME).dt.total_seconds() / 3600
    c["ASSESSED_ICU"] = c.FIRST_NOTE.notna()
    for h in (24, 48, 72):
        c[f"WITHIN_{h}H"] = c.HOURS_TO_NOTE <= h
    c["NOTES_PER_7D"] = c.N_NOTES_ICU / (c.LOS_H / 168)
    return c


# ------------------------------------------------------------------- summaries
def q(v, d=1):
    v = pd.Series(v).dropna()
    if len(v) == 0:
        return ""
    return f"{v.median():.{d}f} ({v.quantile(.25):.{d}f}-{v.quantile(.75):.{d}f})"


def pct_ci(k, n):
    lo, hi = m3.wilson(k, n)
    return f"{100 * k / n:.1f} ({100 * lo:.1f}-{100 * hi:.1f})"


def by_group(c):
    return [("All", c)] + [(g, c[c.GROUP == g]) for g in GROUPS]


def table1(c):
    rows = []
    for grp, s in by_group(c):
        n = len(s)
        if n < SMALL:
            rows.append({"group": grp, "n": m3.suppress(n)})
            continue
        r = {"group": grp, "n": n, "age_median_iqr": q(s.AGE),
             "female_pct": round(100 * (s.GENDER == "F").mean(), 1),
             "invasive_ventilation_48h_pct": round(100 * s.VENT48.mean(), 1),
             "emergency_admission_pct": round(100 * (s.ADMISSION_TYPE == "EMERGENCY").mean(), 1),
             "icu_los_days_median_iqr": q(s.LOS),
             "hospital_mortality_pct": round(100 * s.HOSPITAL_EXPIRE_FLAG.mean(), 1)}
        for cu, k in s.FIRST_CAREUNIT.value_counts().items():
            r[f"unit_{cu}_pct"] = round(100 * k / n, 1)
        rows.append(r)
    return pd.DataFrame(rows)


def assessment_summary(c):
    rows = []
    for grp, s in by_group(c):
        n = len(s)
        if n < SMALL:
            rows.append({"group": grp, "n": m3.suppress(n)})
            continue
        a = s[s.ASSESSED_ICU]
        rows.append({
            "group": grp, "n": n,
            "assessed_during_icu_pct_ci": pct_ci(int(s.ASSESSED_ICU.sum()), n),
            "within_24h_pct_ci": pct_ci(int(s.WITHIN_24H.sum()), n),
            "within_48h_pct_ci": pct_ci(int(s.WITHIN_48H.sum()), n),
            "within_72h_pct_ci": pct_ci(int(s.WITHIN_72H.sum()), n),
            "hours_to_first_median_iqr_assessed": q(a.HOURS_TO_NOTE) if len(a) >= SMALL else "",
            "notes_per_7_icu_days_median_iqr": q(s.NOTES_PER_7D),
            "pre_icu_assessment_pct": round(100 * s.PRE_ICU_NOTE.mean(), 1)})
    return pd.DataFrame(rows)


def cumulative_incidence(c, hours=CIF_HOURS):
    """Cumulative incidence of first assessment, with ICU discharge or death before
    assessment as a competing event. Every stay ends (no censoring), so the
    Aalen-Johansen estimator equals the empirical proportion assessed by time t."""
    rows = []
    for grp, s in by_group(c):
        n = len(s)
        if n < SMALL:
            continue
        t_assess = s.HOURS_TO_NOTE.where(s.ASSESSED_ICU, np.inf)
        for h in hours:
            k = int((t_assess <= h).sum())
            lo, hi = m3.wilson(k, n)
            rows.append({"group": grp, "hour": h, "n": n,
                         "cum_incidence_pct": round(100 * k / n, 1),
                         "ci_low": round(100 * lo, 1), "ci_high": round(100 * hi, 1),
                         "still_in_icu_unassessed": m3.suppress(int(((t_assess > h) & (s.LOS_H > h)).sum()))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ regression
def logit_robust(y, X, iters=50):
    """Logistic regression by IRLS with HC0 sandwich standard errors."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = p * (1 - p)
        H = (X.T * W) @ X
        step = np.linalg.pinv(H) @ (X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    Hinv = np.linalg.pinv((X.T * (p * (1 - p))) @ X)
    meat = (X.T * (y - p) ** 2) @ X
    se = np.sqrt(np.diag(Hinv @ meat @ Hinv))
    z = b / se
    pv = 2 * stats.norm.sf(np.abs(z))
    return b, se, pv


def regression(c, outcome="WITHIN_48H"):
    d = c.dropna(subset=["AGE"]).copy()
    X = pd.DataFrame({"Intercept": 1.0}, index=d.index)
    for g in GROUPS[1:]:
        X[f"Group: {g} (vs no kidney disease)"] = (d.GROUP == g).astype(float)
    X["Age, per 10 years"] = d.AGE / 10
    X["Female sex"] = (d.GENDER == "F").astype(float)
    X["Invasive ventilation in first 48 h"] = d.VENT48.astype(float)
    X["Emergency admission"] = (d.ADMISSION_TYPE == "EMERGENCY").astype(float)
    for cu in sorted(d.FIRST_CAREUNIT.dropna().unique()):
        if cu != "MICU":
            X[f"First unit: {cu} (vs MICU)"] = (d.FIRST_CAREUNIT == cu).astype(float)
    keep = [col for col in X if col == "Intercept" or X[col].std() > 0]
    X = X[keep]
    b, se, p = logit_robust(d[outcome].astype(float), X)
    out = pd.DataFrame({"term": X.columns, "odds_ratio": np.exp(b),
                        "ci_low": np.exp(b - 1.96 * se), "ci_high": np.exp(b + 1.96 * se), "p": p})
    out = out[out.term != "Intercept"]
    out[["odds_ratio", "ci_low", "ci_high"]] = out[["odds_ratio", "ci_low", "ci_high"]].round(2)
    out["p"] = out.p.map(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
    out["n"] = m3.suppress(len(d))
    out["events"] = m3.suppress(int(d[outcome].sum()))
    return out


# ------------------------------------------------------------ link to EN start
def first_en_start(rows_amounts):
    en = rows_amounts[rows_amounts.type.eq("EN") & (rows_amounts.EN_ML > 0)]
    return en.groupby("ICUSTAY_ID").STARTTIME.min().rename("FIRST_EN").reset_index()


def en_link(c):
    rows = []
    for grp, s in by_group(c):
        fed = s[s.FIRST_EN.notna()]
        n = len(fed)
        if n < SMALL:
            rows.append({"group": grp, "n_received_en": m3.suppress(n)})
            continue
        before = fed.ASSESSED_ICU & (fed.FIRST_NOTE <= fed.FIRST_EN)
        gap = (fed.FIRST_EN - fed.FIRST_NOTE).dt.total_seconds() / 3600
        rows.append({
            "group": grp, "n_received_en": n,
            "en_received_pct_of_group": round(100 * n / len(s), 1),
            "hours_icu_to_first_en_median_iqr": q((fed.FIRST_EN - fed.INTIME).dt.total_seconds() / 3600),
            "assessed_before_en_start_pct": round(100 * before.mean(), 1),
            "assessed_after_en_start_pct": round(100 * (fed.ASSESSED_ICU & ~before).mean(), 1),
            "not_assessed_in_icu_pct": round(100 * (~fed.ASSESSED_ICU).mean(), 1),
            "hours_assessment_to_en_median_iqr": q(gap[before])})
    return pd.DataFrame(rows)


# --------------------------------------- link: early assessment and delivery
EARLY_H = 72          # assessed before the primary delivery window (ICU days 4-7) starts


def link_assessment_delivery(final, means, notes, aspen_low=1.2):
    """Protein and energy delivery on days 4-7 by dietitian assessment in the first 72 h."""
    a = assessment_by_stay(final, notes)
    d = final.merge(a, on="ICUSTAY_ID", how="left").merge(
        means[["ICUSTAY_ID", "PROTEIN_GKG", "KCAL_KG"]], on="ICUSTAY_ID", how="left")
    h = (d.FIRST_NOTE - d.INTIME).dt.total_seconds() / 3600
    d["EARLY_ASSESS"] = (h <= EARLY_H).fillna(False).astype(bool)
    rows = []
    for grp, s in [("All", d)] + [(g, d[d.GROUP == g]) for g in GROUPS]:
        for early, t in [(True, s[s.EARLY_ASSESS]), (False, s[~s.EARLY_ASSESS])]:
            n = len(t)
            r = {"group": grp, "assessed_within_72h": early, "n": m3.suppress(n)}
            if n >= SMALL:
                k = int((t.PROTEIN_GKG >= 0.8 * aspen_low).sum())
                r.update({"protein_gkg_median_iqr": q(t.PROTEIN_GKG, 2),
                          "kcal_kg_median_iqr": q(t.KCAL_KG, 1),
                          "reach80_aspen_protein_pct_ci": pct_ci(k, n)})
            rows.append(r)
    return d, pd.DataFrame(rows)
