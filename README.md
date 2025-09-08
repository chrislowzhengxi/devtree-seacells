# devtree-seacells

Single-cell analysis of embryonic development using **SEACells** and **UCell**.  
This repository contains code, notebooks, and selected figures from analyses performed on the **Shendure developmental tree dataset**, spanning multiple organ systems.

---

## 📌 Overview
- **SEACells**: identifies metacells (pseudobulk units) that capture gene expression structure across large scRNA-seq datasets.
- **UCell**: computes per-cell enrichment scores for gene signatures, such as Hedgehog signaling (GLI1, PTCH1, HHIP, FOXF1, FOXF2).
- **Systems analyzed**: Renal, Gut, Lateral plate mesoderm, PNS neurons, and others.
- **Figures**: Selected PDFs/PNGs summarizing QC and system-specific results (large raw data excluded).

---

## 📂 Repository structure
```
code/                 # Python scripts for SEACells, UCell scoring, plotting
notebooks/            # Analysis notebooks (Midway3 HPC)
results/              # Figures (PDF/PNG/SVG) and selected QC tables
  └─ system_qc/       # QC plots by system
  └─ ucell/           # UCell QC tables (kept small)
```

---

## 🚀 Getting started

### Environment
Create the conda environment:
```bash
mamba env create -f environment.yml
conda activate scvitools   # or your env name
```

### Running analyses
Example (SEACells on the Eye dataset):
```bash
python code/SEACells_eye.py --adata data/eye.h5ad --out results/eye_qc/
```

---

## 📊 Data
- Input `.h5ad` files are **not included** (too large).  
- Original datasets can be found at: [GEO accession GSE267719](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE267719).  
- See `DATA.md` for instructions on locating processed `.h5ad` files on Midway3.

---

## 🔍 Notes
- Large intermediate outputs (`.h5ad`, raw `.csv`) are excluded via `.gitignore`.  
- Only small, essential QC tables and final figures are tracked.  
- Figures with `testp` in their name are excluded to keep the repo clean.

---

## ✨ Credits
- Analysis by Chris Low (University of Chicago).  
- Methods:  
  - SEACells (Cortal et al., Nat Biotech 2023)
  - UCell (Andreatta & Carmona, Bioinformatics 2021)
- Data: Shendure lab (GSE267719).
