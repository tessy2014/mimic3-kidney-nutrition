# Dietitian assessment and protein–energy delivery in critically ill patients with kidney disease (MIMIC-III)

Analysis code for the manuscript:

> Badriyah T, Faradisa R. *Dietitian assessment, timing of enteral nutrition and protein–energy delivery in critically ill patients with kidney disease: a retrospective cohort study.* Submitted to Clinical Nutrition ESPEN.

The code reproduces every table and figure of the manuscript from MIMIC-III v1.4 (MetaVision, 2008–2012).

## No patient data in this repository

MIMIC-III is available only to credentialed users under the PhysioNet Credentialed Health Data Use Agreement:
<https://physionet.org/content/mimiciii/1.4/>. This repository contains **code and reference tables only**.
Do not add MIMIC files, patient-level outputs or screenshots of rows to this repository or to any online service.
Clinical note text is never read; only the time of Nutrition notes is used.

## Contents

| File | Purpose |
|---|---|
| `mimic_t1_dietitian_assessment.ipynb` | Dietitian assessment (frequency, timing), start of enteral nutrition, cumulative incidence, logistic regression, link between early assessment and delivery |
| `mimic_t3_protein_energy.ipynb` | Protein and energy delivery on ICU days 1–7, comparison with ASPEN/SCCM and ESPEN targets, linear regression, sensitivity analyses |
| `mimic_t1.py`, `mimic_t3.py` | Functions used by the notebooks |
| `formula_composition.csv` | Energy and protein per mL (or per g) of each enteral formula, modular protein, propofol and PN component, with sources and verification status |
| `guideline_targets.csv` | ASPEN/SCCM and ESPEN protein and energy ranges by kidney group, with references |

## How to run

1. Obtain MIMIC-III v1.4 access on PhysioNet and download: `ICUSTAYS`, `PATIENTS`, `ADMISSIONS`, `DIAGNOSES_ICD`,
   `PROCEDURES_ICD`, `PROCEDUREEVENTS_MV`, `INPUTEVENTS_MV`, `D_ITEMS`, `NOTEEVENTS` (`.csv.gz`).
2. Put the MIMIC files and all files of this repository in one folder.
3. Set the folder path, then start Jupyter:

   ```bash
   pip install -r requirements.txt
   export MIMIC_DIR=/path/to/folder      # Windows PowerShell: $env:MIMIC_DIR="C:\path\to\folder"
   jupyter lab
   ```
4. Run `mimic_t3_protein_energy.ipynb` and `mimic_t1_dietitian_assessment.ipynb` with *Kernel → Restart Kernel and Run All Cells*.

Results are written to `outputs/`. Only files starting with `summary_` contain aggregate numbers (counts of 1–10 shown as `<11`)
and may be shared; all other files in `outputs/` stay on your own computer.

## Methods in brief

- Adults at their first MetaVision ICU stay. Assessment cohort: ICU stay ≥48 h. Delivery cohort: ICU stay ≥96 h, enteral nutrition on days 1–7, no parenteral nutrition, weight 30–250 kg.
- Kidney groups (hierarchical): continuous KRT, intermittent KRT, kidney disease without KRT (ICD-9 584, 585, 403, 404), no kidney disease.
- Infusions are split across 24-h ICU days in proportion to running time; only complete days are analysed.
- Targets from ASPEN/SCCM (2016, 2022) and ESPEN (ICU 2019/2023; kidney disease 2021, Recommendation 18).

## Citation

If you use this code, please cite the archived release (see `CITATION.cff`; DOI from Zenodo) and the article once published.

## Licence

Code: MIT licence (see `LICENSE`). Reference tables: CC BY 4.0.
