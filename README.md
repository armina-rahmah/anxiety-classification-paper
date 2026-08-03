# Psychometric-Aware Anxiety-Severity Classification from HARS Item Responses

Reproducible code for a psychometric-aware stacking framework that estimates
five-level Hamilton Anxiety Rating Scale (HARS) severity from item-level
responses. The pipeline models the 14 HARS items as ordinal responses using
tabular item-response descriptors and an ordered item sequence, and fuses
Random Forest, SVM, and LSTM base learners through a logistic-regression
meta-learner trained on out-of-fold class probabilities. The aggregate/total
score is used **only** to construct the severity label and is never used as a
predictive feature.

## Repository contents

| File | Description |
|------|-------------|
| `HARS_pipeline_github_release.ipynb` | End-to-end pipeline on synthetic data: data generation, label construction, feature design, model comparison, stacking, ablation, robustness, repeated cross-validation, multi-split evaluation, and figures. |
| `README.md` | This file. |
| `Results/` | Precomputed result tables (CSV) and figures from a full run. |

> The synthetic dataset is generated inside the notebook itself (a seeded
> generator that replaces the private Excel file); no separate data file is
> required to run it.

## Data availability

The original questionnaire responses contain sensitive student mental-health
data and identifiers and are therefore **not** shared. The notebook instead
generates a fully **synthetic** dataset with the same schema (14 ordinal items
scored 0-4) and a comparable five-class severity distribution, so the entire
pipeline can be run end-to-end and every reported table reproduced without any
real participant data. The modelling code is identical to the version used on
the real data; only the data-loading section differs.

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `tensorflow`,
  `xgboost`, `lightgbm`, `catboost`, `mord`, `openpyxl`, `matplotlib`

Install:

```bash
pip install numpy pandas scikit-learn imbalanced-learn tensorflow xgboost lightgbm catboost mord openpyxl matplotlib
```

## How to run

### Option 1 - Google Colab (recommended)

1. Upload `HARS_pipeline_github_release.ipynb` to Colab.
2. Run the first cell to install the dependencies.
3. Select **Runtime -> Run all**. No data file or Google Drive mount is required;
   the synthetic dataset is generated automatically. All tables and figures are
   produced in order and written to `./github_release_outputs`.

### Option 2 - Local

```bash
jupyter notebook HARS_pipeline_github_release.ipynb
```

Run all cells. Outputs are written to `./github_release_outputs`.

> To run the same pipeline on your own data instead of the synthetic set,
> replace the synthetic-generation cell with a loader that produces a data frame
> of 14 ordinal item columns (values 0-4) named `Q1_...`-`Q14_...`; the rest of
> the notebook is unchanged.

## Reproducibility

- A fixed random seed (`RANDOM_STATE = 42`) is set for NumPy, Python, and
  TensorFlow, so a top-to-bottom run reproduces the reported results.
- The train/test split, five-fold internal cross-validation, SMOTE (applied to
  training folds only), and all model hyperparameters are defined explicitly in
  the notebook.
- **Robustness with multiple seeds:** the noisy-response experiment is repeated
  over `N_NOISE_SEEDS = 10` perturbation seeds; the proposed model is
  re-estimated over `N_SEEDS_PROPOSED = 5` independent stratified splits; the
  lightweight competitors are evaluated with repeated stratified 5x10-fold
  cross-validation (`N_REPEATS_CV = 10`).
- **Interval reporting:** the resampling-based panels (repeated cross-validation
  and multi-split) report the **empirical 2.5th-97.5th percentile interval** of
  the observed performance distribution; the multi-seed robustness summaries
  additionally report **mean +/- standard deviation** and a **t-based 95%
  confidence interval for the mean**. The two evaluation panels use different
  resampling protocols and are reported separately, not ranked directly against
  each other.

## Outputs

Running the notebook writes the following to the output directory:
`main_performance_table.csv`, `per_class_recall_table.csv`,
`feature_ablation_table.csv`, `feature_representation_summary.csv`,
`missing_item_robustness_table.csv`, `noisy_response_robustness_table.csv`,
`noisy_response_multiseed_all_runs.csv`, `repeated_cv_summary.csv`,
`table10_panelA_repeated_cv.csv`, `table10_panelB_proposed_multisplit.csv`,
`proposed_multiseed_summary.csv`, `rule_vs_model_degradation.csv`,
`reliability_analysis.csv`, `class_distribution.csv`, and the corresponding
figures (`overall_performance_comparison.png`, `brier_score_comparison.png`,
`confusion_matrix_stacking.png`, `missing_item_robustness.png`,
`noisy_response_robustness.png`).

## Citation

If you use this code, please cite the associated article (details to be added
upon publication).

## License

Released under the MIT License.
