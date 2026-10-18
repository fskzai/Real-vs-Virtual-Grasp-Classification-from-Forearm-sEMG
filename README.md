# Real vs Virtual Grasp Classification from Multi-Muscle Forearm sEMG

Machine-learning pipeline that identifies **real-world and virtual-reality (VR) grasping** from surface electromyography (sEMG) of three forearm muscles. It covers six classic classifiers and several newer techniques (Riemannian geometry, subject normalisation, gradient boosting, MiniRocket, a 1D-CNN and a soft-voting ensemble), all evaluated with **subject-independent Leave-One-Subject-Out (LOSO)** validation.


## Overview

| | |
|---|---|
| **Task** | 5-class classification: Rest, Real Ball, Virtual Ball, Real Remote, Virtual Remote |
| **Participants** | 12 healthy adults (6 M / 6 F, 19–24 y) |
| **Muscles** | Flexor Pollicis Longus (FPL), Flexor Digitorum Profundus (FDP), Flexor Digitorum Superficialis (FDS) |
| **Hardware** | BIOPAC MP36, 2 kHz sampling; Oculus Quest 2 for the VR tasks |
| **Protocol** | 6 × (5 s grasp + 5 s rest) per object and environment |
| **Validation** | LOSO (primary) and 20-fold trial-grouped CV (secondary) |

## Pipeline

```
.acq recordings ─► band-pass 20–450 Hz + 50 Hz notch ─► grasp-onset detection
      ─► 1 s windows, 0.5 s step (7 windows per 5 s trial)
      ─► features ──┬─ 20 time/frequency features per channel (article set)
                    ├─ extended TD / AR(4) / Hjorth / band-power + cross-channel features
                    └─ filter-bank Riemannian covariance (5 bands) ─► tangent space
      ─► classifiers (scaling + SMOTE inside training folds only)
      ─► LOSO / 20-fold evaluation ─► tables, confusion matrices, figures, saved models
```

### Classifiers

**Article baselines:** K-Nearest Neighbours, Decision Tree, Gaussian Naive Bayes, Logistic Regression, Multilayer Perceptron, Support Vector Classifier.

**Newer techniques:**
- **Unsupervised subject normalisation.** Each subject's features are z-scored using only their own *unlabelled* windows, so no calibration labels or MVC trials are needed.
- **Filter-bank Riemannian geometry.** Channel covariance matrices are computed in 5 frequency bands, re-centred per subject, mapped to tangent space and classified with an SVM.
- **LightGBM** gradient boosting on the extended feature set.
- **Hybrid fusion** (handcrafted + Riemannian features) and a **hierarchical classifier** (rest → environment → object).
- **MiniRocket**, a multivariate time-series transform with a ridge classifier.
- **1D-CNN with squeeze-excitation** on the raw signal, with channel-gain augmentation.
- **Soft-voting ensemble** across the model families.

## Results (LOSO, 12 subjects)

| Model | Test accuracy (%) | Macro-F1 (%) |
|---|---|---|
| SVC + SMOTE (article baseline) | 85.3 | 76.5 |
| Subject-normalised SVC | 88.6 | 81.8 |
| Riemannian tangent space + SVC | 89.4 | 83.2 |
| **Soft-voting ensemble** | **90.6** | **85.0** |

- The ensemble improved macro-F1 over the baseline in 11 of 12 subjects (Wilcoxon signed-rank p ≈ 0.002).
- Rest vs grasp and real vs virtual are separated almost perfectly. **Virtual Ball vs Virtual Remote** is the hardest distinction (≈ 66–70% recall).
- Fusing all three muscles beats any single muscle or pair.

> Results come from the authors' run of the notebook. Re-running reproduces them closely; the CNN varies slightly between runs, and GPU runs may differ.

## Repository structure

```
├── sEMG_Training_Notebook.ipynb   # complete training + evaluation notebook
├── README.md
└── outputs/                       # created when the notebook runs
    ├── results_tables.xlsx        # LOSO / 20-fold tables + per-fold results
    ├── results_all.csv
    ├── muscle_configurations.csv
    ├── wilcoxon_vs_article_SVC.csv
    ├── figures/                   # 16 figures at 300 dpi (confusion matrices, ROC, t-SNE, …)
    └── trained_models/            # final models (.joblib, .pt) + preprocessing.json
```

## Getting started

### 1. Environment
```bash
conda create -n semg python=3.10 -y
conda activate semg
pip install bioread imbalanced-learn pyriemann lightgbm aeon torch scikit-learn scipy pandas matplotlib seaborn joblib openpyxl ipykernel
python -m ipykernel install --user --name semg
```
> If you reuse an existing environment and see *"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"*, run `pip install "numpy<2"` or upgrade PyTorch.

### 2. Data
Place the BIOPAC `.acq` files in one folder (sub-folders are fine). File names must contain the **subject number**, the **object** (`ball` / `remote`) and `vr` for virtual trials, for example `3 ball.acq`, `3 remote vr.acq`, `10 vr ball.acq`.

### 3. Run
Open `sEMG_Training_Notebook.ipynb`, set `DATA_DIR` in the configuration cell, then **Run All**.

| Setting | Default | Purpose |
|---|---|---|
| `CV_SCHEMES` | `["LOSO", "20-fold"]` | validation schemes to run |
| `RUN_CNN` / `RUN_MINIROCKET` | `True` | turn the deep / time-series models on or off |
| `DEEP_ON_KFOLD` | `False` | also run CNN / MiniRocket under 20-fold |
| `RUN_MUSCLE_ANALYSIS` | `True` | single, dual and triple muscle configurations |
| `TUNE_HYPERPARAMS` | `False` | inner GridSearchCV grouped by subject |

Expect about 1–1.5 h on a laptop CPU. With a CUDA GPU the CNN finishes in minutes.

## Methodological safeguards

- Scaling, SMOTE, hyperparameter search and model fitting happen **only inside training folds**. Test folds stay untouched and keep their natural class imbalance.
- All windows from one trial stay in the same fold, so overlapping windows never leak between training and test sets.
- Subject normalisation and Riemannian re-centring use unlabelled data only, never test labels.
- Metrics are macro-averaged. Trial-level results (averaging the 7 windows of each grasp) are reported alongside window-level results.

## Data availability

The sEMG dataset is available from the corresponding author on reasonable request (drzia@au.edu.pk).

**Notes on the data:**
- Two recordings in the shared dataset are empty (subject 6, *remote* and *VR remote*), so 46 of 48 recordings are used.
- Channel 3 is labelled `EDM` inside the `.acq` files. Set `CH_NAMES` in the notebook to match the muscle actually recorded.

## Citation

If you use this code, please cite the accompanying article (citation details will be added on publication):

```bibtex
@article{ajaz_real_virtual_grasp_semg,
  title   = {Machine Learning-Based Identification of Real and Virtual Grasping Using Individual and Fused Multi-Muscle Forearm sEMG Signals},
  author  = {Ajaz, Nishwa and Mohy Ud Din, Zia and Fatima, Noor and Leong, Wai Yie},
  journal = {to appear},
  year    = {2026}
}
```

## License

Released under the MIT License (add a `LICENSE` file to the repository).

## Acknowledgements

Department of Biomedical Engineering, Air University, Islamabad, and the Faculty of Engineering and Technology, INTI International University, Malaysia. Thanks to all participants.
