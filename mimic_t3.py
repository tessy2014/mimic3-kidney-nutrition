"""
Protein and energy delivery versus ASPEN/SCCM and ESPEN targets in critically ill
patients with acute or chronic kidney disease (MIMIC-III v1.4, MetaVision era).

All functions run locally. Only objects written by save_summary() (files named
summary_*) contain aggregate counts and may be shared. Patient-level files are
written with the prefix local_ and must stay on the credentialed user's computer.
"""
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_DAYS = 7
WINDOW = (4, 7)          # primary window: ICU days 4-7 (after the early acute phase)
MIN_LOS_DAYS = 4         # ICU stay of at least 96 h, so every stay has a full ICU day 4
SMALL = 11               # counts 1-10 are shown as "<11" in shareable files
VALUE_COLS = ["KCAL", "PROTEIN", "KCAL_PROPOFOL", "KCAL_PN", "PROTEIN_PN", "EN_ML", "PN_ML"]
GROUPS = ["No kidney disease", "Kidney disease without KRT", "Intermittent KRT", "CKRT"]

ESRD_CODES = {"5856", "40301", "40311", "40391", "40402", "40403", "40412",
              "40413", "40492", "40493", "V4511"}
KRT_PROC_CODES = {"3995", "5498"}          # haemodialysis, peritoneal dialysis


# ----------------------------------------------------------------------------- io
def rd(path, **kw):
    df = pd.read_csv(path, **kw)
    df.columns = [c.upper() for c in df.columns]
    return df


def suppress(n):
    """Small-cell suppression for shareable tables."""
    if pd.isna(n):
        return n
    n = int(n)
    return f"<{SMALL}" if 0 < n < SMALL else n


def save_summary(df, out_dir, name):
    """Write a shareable aggregate table. name must start with 'summary_'."""
    assert name.startswith("summary_"), "shareable files must start with summary_"
    p = Path(out_dir) / f"{name}.csv"
    df.to_csv(p, index=False)
    return p


# ------------------------------------------------------------------------- cohort
def load_icustays(d):
    d = Path(d)
    icu = rd(d / "ICUSTAYS.csv.gz", parse_dates=["INTIME", "OUTTIME"])
    pat = rd(d / "PATIENTS.csv.gz", usecols=["SUBJECT_ID", "GENDER", "DOB"],
             parse_dates=["DOB"])
    adm = rd(d / "ADMISSIONS.csv.gz",
             usecols=["HADM_ID", "ADMISSION_TYPE", "HOSPITAL_EXPIRE_FLAG"])
    df = icu.merge(pat, on="SUBJECT_ID").merge(adm, on="HADM_ID")
    before_bday = ((df.INTIME.dt.month * 100 + df.INTIME.dt.day)
                   < (df.DOB.dt.month * 100 + df.DOB.dt.day)).astype(int)
    age = df.INTIME.dt.year - df.DOB.dt.year - before_bday
    df["AGE"] = np.where(age >= 200, 91.4, age)      # ages >89 are shifted in MIMIC-III
    return df


def base_cohort(icu):
    """Apply the stay-level criteria; returns cohort and flow counts."""
    flow = [("All ICU stays in MIMIC-III", len(icu), icu.SUBJECT_ID.nunique())]
    c = icu[icu.DBSOURCE.str.lower() == "metavision"]
    flow.append(("MetaVision stays (2008-2012)", len(c), c.SUBJECT_ID.nunique()))
    c = c[c.AGE >= 18]
    flow.append(("Age 18 years or older", len(c), c.SUBJECT_ID.nunique()))
    c = c.sort_values("INTIME").drop_duplicates("SUBJECT_ID", keep="first")
    flow.append(("First MetaVision ICU stay per patient", len(c), c.SUBJECT_ID.nunique()))
    c = c[c.LOS >= MIN_LOS_DAYS]
    flow.append((f"ICU stay {MIN_LOS_DAYS * 24} h or longer", len(c), c.SUBJECT_ID.nunique()))
    return c.reset_index(drop=True), flow


# ----------------------------------------------------------------- kidney groups
def kidney_dx(d, hadm_ids):
    dx = rd(Path(d) / "DIAGNOSES_ICD.csv.gz", usecols=["HADM_ID", "ICD9_CODE"],
            dtype={"ICD9_CODE": str})
    dx = dx[dx.HADM_ID.isin(hadm_ids)].copy()
    c = dx.ICD9_CODE.fillna("")
    dx["AKI"] = c.str.startswith("584")
    dx["CKD"] = c.str.startswith(("585", "403", "404"))
    dx["ESRD"] = c.isin(ESRD_CODES)
    out = dx.groupby("HADM_ID")[["AKI", "CKD", "ESRD"]].any().reset_index()
    pr = rd(Path(d) / "PROCEDURES_ICD.csv.gz", usecols=["HADM_ID", "ICD9_CODE"],
            dtype={"ICD9_CODE": str})
    pr = pr[pr.HADM_ID.isin(hadm_ids) & pr.ICD9_CODE.isin(KRT_PROC_CODES)]
    out["KRT_ICD"] = out.HADM_ID.isin(pr.HADM_ID)
    return out


def procedure_flags(d, cohort, hours=24 * N_DAYS):
    """Timed KRT and invasive ventilation from PROCEDUREEVENTS_MV, from 24 h before ICU
    admission to `hours` after it (default: end of ICU day 7)."""
    d = Path(d)
    items = rd(d / "D_ITEMS.csv.gz", usecols=["ITEMID", "LABEL", "LINKSTO"])
    items = items[items.LINKSTO.str.lower() == "procedureevents_mv"]
    lab = items.LABEL.str.lower().fillna("")
    crrt_ids = set(items.ITEMID[lab.str.contains(r"crrt|cvvh|scuf")])
    ihd_ids = set(items.ITEMID[lab.str.contains(r"hemodialysis|peritoneal dialysis")
                               & ~lab.str.contains(r"crrt|cvvh|scuf")])
    vent_ids = set(items.ITEMID[lab.str.contains(r"invasive ventilation")
                                & ~lab.str.contains(r"non-invasive|noninvasive")])
    pe = rd(d / "PROCEDUREEVENTS_MV.csv.gz",
            usecols=["ICUSTAY_ID", "ITEMID", "STARTTIME", "STATUSDESCRIPTION"],
            parse_dates=["STARTTIME"])
    pe = pe[pe.ICUSTAY_ID.isin(cohort.ICUSTAY_ID)
            & (pe.STATUSDESCRIPTION.fillna("") != "Rewritten")]
    pe = pe.merge(cohort[["ICUSTAY_ID", "INTIME"]], on="ICUSTAY_ID")
    h = (pe.STARTTIME - pe.INTIME).dt.total_seconds() / 3600
    pe = pe[(h >= -24) & (h < hours)]
    f = pd.DataFrame({"ICUSTAY_ID": cohort.ICUSTAY_ID})
    f["CKRT_EVT"] = f.ICUSTAY_ID.isin(pe.ICUSTAY_ID[pe.ITEMID.isin(crrt_ids)])
    f["IKRT_EVT"] = f.ICUSTAY_ID.isin(pe.ICUSTAY_ID[pe.ITEMID.isin(ihd_ids)])
    f["VENT"] = f.ICUSTAY_ID.isin(pe.ICUSTAY_ID[pe.ITEMID.isin(vent_ids)])
    labels = items[items.ITEMID.isin(crrt_ids | ihd_ids | vent_ids)][["ITEMID", "LABEL"]]
    return f, labels


def assign_group(df):
    g = np.select(
        [df.CKRT_EVT,
         df.IKRT_EVT | (df.ESRD & df.KRT_ICD),
         df.AKI | df.CKD],
        ["CKRT", "Intermittent KRT", "Kidney disease without KRT"],
        default="No kidney disease")
    sub = np.select([df.CKD | df.ESRD, df.AKI], ["CKD (with or without AKI)", "AKI only"],
                    default="")
    df = df.copy()
    df["GROUP"] = pd.Categorical(g, categories=GROUPS)
    df["SUBGROUP"] = np.where(df.GROUP == "Kidney disease without KRT", sub, "")
    return df


# ------------------------------------------------------------------ nutrition items
def nutrition_items(d, comp):
    """Candidate inputevents_mv items and their mapping to the composition table."""
    items = rd(Path(d) / "D_ITEMS.csv.gz", usecols=["ITEMID", "LABEL", "CATEGORY", "LINKSTO"])
    items = items[items.LINKSTO.str.lower() == "inputevents_mv"].copy()
    lab = items.LABEL.fillna("").str.lower()
    cat = items.CATEGORY.fillna("").str.lower()
    any_pat = "|".join(f"(?:{p})" for p in comp.pattern)
    items = items[cat.str.contains("nutrition") | lab.str.contains(any_pat, regex=True)].copy()

    def match(label):
        s = str(label).lower()
        for _, r in comp.iterrows():
            if re.search(r.pattern, s):
                return r.name
        return np.nan

    items["COMP_IDX"] = items.LABEL.map(match)
    comp2 = comp.reset_index().rename(columns={"index": "COMP_IDX"})
    items = items.merge(comp2, on="COMP_IDX", how="left")
    s = items.LABEL.fillna("").str.lower().str.extract(r"\((full|3/4|2/3|1/2|1/3|1/4)\)")[0]
    frac = {"full": 1.0, "3/4": 0.75, "2/3": 2 / 3, "1/2": 0.5, "1/3": 1 / 3, "1/4": 0.25}
    items["STRENGTH"] = s.map(frac).fillna(1.0)
    return items


def scan_inputevents(d, cohort, items, chunksize=1_000_000):
    """One chunked pass: nutrition rows of cohort stays plus first recorded weight."""
    keep_ids = set(items.ITEMID)
    stays = set(cohort.ICUSTAY_ID)
    cols = ["ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "AMOUNT", "AMOUNTUOM",
            "PATIENTWEIGHT", "STATUSDESCRIPTION"]
    rows, wts = [], []
    for ch in pd.read_csv(Path(d) / "INPUTEVENTS_MV.csv.gz", usecols=cols, chunksize=chunksize,
                          parse_dates=["STARTTIME", "ENDTIME"]):
        ch = ch[ch.ICUSTAY_ID.isin(stays)]
        ch = ch[ch.STATUSDESCRIPTION.fillna("") != "Rewritten"]
        w = ch.dropna(subset=["PATIENTWEIGHT"]).sort_values("STARTTIME")
        wts.append(w.groupby("ICUSTAY_ID").first()[["STARTTIME", "PATIENTWEIGHT"]].reset_index())
        rows.append(ch[ch.ITEMID.isin(keep_ids)])
    rows = pd.concat(rows, ignore_index=True)
    w = pd.concat(wts).sort_values("STARTTIME").groupby("ICUSTAY_ID").first().reset_index()
    w = w.rename(columns={"PATIENTWEIGHT": "WEIGHT"})[["ICUSTAY_ID", "WEIGHT"]]
    return rows, w


def nutrient_amounts(rows, items):
    """Convert recorded amounts into kcal and protein grams per row."""
    r = rows.merge(items[["ITEMID", "LABEL", "type", "unit", "kcal_per_unit",
                          "protein_g_per_unit", "STRENGTH"]], on="ITEMID", how="left")
    uom = r.AMOUNTUOM.fillna("").str.lower()
    amt = r.AMOUNT.astype(float)
    qty = pd.Series(np.nan, index=r.index)
    ml_unit = r.unit.eq("ml")
    qty[ml_unit & uom.eq("ml")] = amt
    qty[ml_unit & uom.eq("l")] = amt * 1000
    qty[r.type.eq("PROPOFOL") & uom.eq("mg")] = amt / 10.0          # 10 mg per mL
    qty[r.type.eq("PROPOFOL") & uom.eq("mcg")] = amt / 10000.0
    g_unit = r.unit.eq("g")
    qty[g_unit & uom.isin(["g", "gm", "grams"])] = amt
    r["QTY"] = qty
    r["UNIT_OK"] = qty.notna()
    eff = r.QTY * r.STRENGTH
    r["KCAL"] = np.where(r.type.isin(["EN", "MODULAR"]), eff * r.kcal_per_unit, 0.0)
    r["PROTEIN"] = np.where(r.type.isin(["EN", "MODULAR"]), eff * r.protein_g_per_unit, 0.0)
    r["KCAL_PROPOFOL"] = np.where(r.type.eq("PROPOFOL"), r.QTY * r.kcal_per_unit, 0.0)
    # PN components that MetaVision records in grams (amino acids, dextrose, lipids)
    r["KCAL_PN"] = np.where(r.type.eq("PN_COMP"), r.QTY * r.kcal_per_unit, 0.0)
    r["PROTEIN_PN"] = np.where(r.type.eq("PN_COMP"), r.QTY * r.protein_g_per_unit, 0.0)
    r["EN_ML"] = np.where(r.type.eq("EN"), r.QTY, 0.0)
    # PN flag: volume of PN bags, or 1 for each recorded PN component
    r["PN_ML"] = np.where(r.type.eq("PN"), amt.where(uom.eq("ml"), 1.0),
                          np.where(r.type.eq("PN_COMP"), 1.0, 0.0))
    r["COMP_MISSING"] = r.type.isin(["EN", "MODULAR"]) & (
        r.kcal_per_unit.isna() | r.protein_g_per_unit.isna())
    for c in VALUE_COLS:
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0.0)
    return r


def allocate_days(r, cohort):
    """Split each infusion across ICU days in proportion to its running time."""
    r = r.merge(cohort[["ICUSTAY_ID", "INTIME"]], on="ICUSTAY_ID")
    s = ((r.STARTTIME - r.INTIME).dt.total_seconds() / 3600).to_numpy()
    e = ((r.ENDTIME - r.INTIME).dt.total_seconds() / 3600).to_numpy()
    e = np.maximum(e, s)
    dur = e - s
    vals = r[VALUE_COLS].to_numpy()
    parts = []
    for k in range(N_DAYS):
        a, b = 24 * k, 24 * (k + 1)
        ov = np.clip(np.minimum(e, b) - np.maximum(s, a), 0, None)
        frac = np.where(dur > 0, ov / np.where(dur > 0, dur, 1), ((s >= a) & (s < b)).astype(float))
        p = pd.DataFrame(vals * frac[:, None], columns=VALUE_COLS)
        p["ICUSTAY_ID"] = r.ICUSTAY_ID.to_numpy()
        p["DAY"] = k + 1
        parts.append(p)
    return pd.concat(parts).groupby(["ICUSTAY_ID", "DAY"], as_index=False).sum()


def daily_table(cohort, alloc):
    """One row per stay and full ICU day 1-7, with zeros on days without nutrition."""
    grid = cohort[["ICUSTAY_ID", "INTIME", "OUTTIME"]].merge(
        pd.DataFrame({"DAY": range(1, N_DAYS + 1)}), how="cross")
    full = grid.INTIME + pd.to_timedelta(grid.DAY * 24, unit="h") <= grid.OUTTIME
    grid = grid[full][["ICUSTAY_ID", "DAY"]]
    day = grid.merge(alloc, on=["ICUSTAY_ID", "DAY"], how="left").fillna(0.0)
    day = day.merge(cohort[["ICUSTAY_ID", "WEIGHT", "GROUP"]], on="ICUSTAY_ID")
    day["PROTEIN_GKG"] = day.PROTEIN / day.WEIGHT
    day["KCAL_KG"] = day.KCAL / day.WEIGHT
    day["KCAL_TOTAL_KG"] = (day.KCAL + day.KCAL_PROPOFOL) / day.WEIGHT
    day["PROTEIN_ENPN_GKG"] = (day.PROTEIN + day.PROTEIN_PN) / day.WEIGHT     # EN + PN
    day["KCAL_ENPN_KG"] = (day.KCAL + day.KCAL_PN) / day.WEIGHT
    day["FED"] = day.EN_ML > 0
    return day


# ---------------------------------------------------------------------- analyses
def window_means(day, window=WINDOW, fed_only=False):
    w = day[(day.DAY >= window[0]) & (day.DAY <= window[1])]
    if fed_only:
        w = w[w.FED]
    return w.groupby("ICUSTAY_ID", as_index=False).agg(
        PROTEIN_GKG=("PROTEIN_GKG", "mean"), KCAL_KG=("KCAL_KG", "mean"),
        KCAL_TOTAL_KG=("KCAL_TOTAL_KG", "mean"), PROTEIN_ENPN_GKG=("PROTEIN_ENPN_GKG", "mean"),
        KCAL_ENPN_KG=("KCAL_ENPN_KG", "mean"), N_DAYS=("DAY", "size"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def adequacy(means, cohort, targets, energy_col="KCAL_KG", group_col="GROUP", target_map=None):
    """Share of stays below, within and above each guideline range (window means).
    target_map lets subgroups use the targets of a parent group."""
    m = means.merge(cohort[["ICUSTAY_ID", group_col]], on="ICUSTAY_ID")
    if target_map:
        targets = pd.concat([targets[targets.group == parent].assign(group=sub)
                             for sub, parent in target_map.items()])
    out = []
    for (gl, grp), t in targets.groupby(["guideline", "group"]):
        sub = m[m[group_col] == grp]
        n = len(sub)
        if n < SMALL:
            out.append({"guideline": gl, "group": grp, "n": n})
            continue
        t = t.iloc[0]
        rec = {"guideline": gl, "group": grp, "n": n}
        for nutr, col, lo, hi in [("protein", "PROTEIN_GKG", t.protein_low, t.protein_high),
                                  ("energy", energy_col, t.kcal_low, t.kcal_high)]:
            v = sub[col]
            below = int((v < lo).sum())
            above = int((v > hi).sum()) if pd.notna(hi) else 0
            within = n - below - above
            reach80 = int((v >= 0.8 * lo).sum())
            lo80, hi80 = wilson(reach80, n)
            rec.update({
                f"{nutr}_target": f"{lo}-{hi}" if pd.notna(hi) else f">={lo}",
                f"{nutr}_below_pct": round(100 * below / n, 1),
                f"{nutr}_within_pct": round(100 * within / n, 1),
                f"{nutr}_above_pct": round(100 * above / n, 1),
                f"{nutr}_reach80_pct": round(100 * reach80 / n, 1),
                f"{nutr}_reach80_ci": f"{100 * lo80:.1f}-{100 * hi80:.1f}"})
        out.append(rec)
    df = pd.DataFrame(out)
    df["n"] = df.n.map(suppress)
    return df


def q(v):
    v = pd.Series(v).dropna()
    if len(v) == 0:
        return ""
    return f"{v.median():.2f} ({v.quantile(.25):.2f}-{v.quantile(.75):.2f})"


def daily_summary(day):
    g = day.groupby(["GROUP", "DAY"], observed=True)
    out = g.agg(n=("ICUSTAY_ID", "size"),
                fed_pct=("FED", lambda x: round(100 * x.mean(), 1)),
                protein_gkg_mean=("PROTEIN_GKG", "mean"),
                protein_gkg_median=("PROTEIN_GKG", "median"),
                protein_gkg_q1=("PROTEIN_GKG", lambda x: x.quantile(.25)),
                protein_gkg_q3=("PROTEIN_GKG", lambda x: x.quantile(.75)),
                kcal_kg_median=("KCAL_KG", "median"),
                kcal_kg_q1=("KCAL_KG", lambda x: x.quantile(.25)),
                kcal_kg_q3=("KCAL_KG", lambda x: x.quantile(.75)),
                kcal_total_kg_median=("KCAL_TOTAL_KG", "median")).reset_index()
    num = out.select_dtypes("float").columns
    out[num] = out[num].round(3)
    out.loc[out.n < SMALL, num] = np.nan          # no statistics for cells with n < 11
    out["n"] = out.n.map(suppress)
    return out


def table1(c, means):
    c = c.merge(means[["ICUSTAY_ID", "PROTEIN_GKG", "KCAL_KG"]], on="ICUSTAY_ID", how="left")
    rows = []
    for grp, s in [("All", c)] + [(g, c[c.GROUP == g]) for g in GROUPS]:
        n = len(s)
        if n < SMALL:
            rows.append({"group": grp, "n": suppress(n)})
            continue
        r = {"group": grp, "n": n,
             "age_median_iqr": q(s.AGE),
             "female_pct": round(100 * (s.GENDER == "F").mean(), 1),
             "weight_kg_median_iqr": q(s.WEIGHT),
             "invasive_ventilation_pct": round(100 * s.VENT.mean(), 1),
             "emergency_admission_pct": round(100 * (s.ADMISSION_TYPE == "EMERGENCY").mean(), 1),
             "icu_los_days_median_iqr": q(s.LOS),
             "hospital_mortality_pct": round(100 * s.HOSPITAL_EXPIRE_FLAG.mean(), 1),
             "first_en_day_median_iqr": q(s.FIRST_EN_DAY),
             "protein_gkg_days4to7_median_iqr": q(s.PROTEIN_GKG),
             "kcal_kg_days4to7_median_iqr": q(s.KCAL_KG)}
        for cu, k in s.FIRST_CAREUNIT.value_counts().items():
            r[f"unit_{cu}_pct"] = round(100 * k / n, 1)
        rows.append(r)
    return pd.DataFrame(rows)


def ols_hc3(y, X):
    """OLS with HC3 robust standard errors (no statsmodels dependency)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (e / (1 - h)) ** 2
    V = XtX_inv @ (X.T * w) @ X @ XtX_inv
    se = np.sqrt(np.diag(V))
    dfree = max(len(y) - X.shape[1], 1)
    t = b / se
    p = 2 * stats.t.sf(np.abs(t), dfree)
    tc = stats.t.ppf(0.975, dfree)
    return b, se, b - tc * se, b + tc * se, p


def regression(c, means, extra=None):
    """Linear regression of mean protein delivery. extra: {label: column of c} added as covariates."""
    d = c.merge(means[["ICUSTAY_ID", "PROTEIN_GKG"]], on="ICUSTAY_ID").dropna(
        subset=["PROTEIN_GKG", "AGE", "WEIGHT"])
    X = pd.DataFrame({"Intercept": 1.0}, index=d.index)
    for label, col in (extra or {}).items():
        X[label] = d[col].astype(float)
    for g in GROUPS[1:]:
        X[f"Group: {g} (vs no kidney disease)"] = (d.GROUP == g).astype(float)
    X["Age, per 10 years"] = d.AGE / 10
    X["Female sex"] = (d.GENDER == "F").astype(float)
    X["Weight, per 10 kg"] = d.WEIGHT / 10
    X["Invasive ventilation"] = d.VENT.astype(float)
    X["Emergency admission"] = (d.ADMISSION_TYPE == "EMERGENCY").astype(float)
    for cu in sorted(d.FIRST_CAREUNIT.dropna().unique()):
        if cu != "MICU":
            X[f"First unit: {cu} (vs MICU)"] = (d.FIRST_CAREUNIT == cu).astype(float)
    X = X.loc[:, X.std() > 0].assign(Intercept=1.0)
    b, se, lo, hi, p = ols_hc3(d.PROTEIN_GKG, X)
    out = pd.DataFrame({"term": X.columns, "beta_gkg": b, "ci_low": lo, "ci_high": hi, "p": p})
    out[["beta_gkg", "ci_low", "ci_high"]] = out[["beta_gkg", "ci_low", "ci_high"]].round(3)
    out["p"] = out.p.map(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
    out.attrs["n"] = len(d)
    return out


def item_inventory(items, r):
    use = r.groupby("ITEMID").agg(n_rows=("ICUSTAY_ID", "size"), n_stays=("ICUSTAY_ID", "nunique"),
                                  unit_ok_pct=("UNIT_OK", lambda x: round(100 * x.mean(), 1)),
                                  top_uom=("AMOUNTUOM", lambda x: x.mode().iat[0] if len(x.mode()) else ""))
    inv = items[["ITEMID", "LABEL", "CATEGORY", "product", "type", "STRENGTH",
                 "kcal_per_unit", "protein_g_per_unit"]].merge(use, on="ITEMID", how="left")
    inv = inv[inv.n_rows.fillna(0) > 0].sort_values("n_rows", ascending=False)
    inv["mapped"] = inv.type.notna()
    inv["n_rows"] = inv.n_rows.map(suppress)
    inv["n_stays"] = inv.n_stays.map(suppress)
    return inv
