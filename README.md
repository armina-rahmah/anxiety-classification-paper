# Psychometric-Aware Anxiety-Severity Classification from HARS Item Responses

Reproducible code for a psychometric-aware stacking framework that estimates
five-level Hamilton Anxiety Rating Scale (HARS) severity from item-level
responses. The pipeline models the 14 HARS items as ordinal responses using
tabular item-response descriptors and an ordered item sequence, and fuses
Random Forest, SVM, and LSTM base learners through a logistic-regression
meta-learner trained on out-of-fold class probabilities.

The aggregate/total score is used **only** to construct the severity label and
is never used as a predictive feature.

## Repository contents

| File | Description |
|------|-------------|
| `HARS_pipeline_release.ipynb` | End-to-end pipeline: data loading, label construction, feature design, model comparison, stacking, ablation, robustness, cross-validation, and figures. |
| `generate_synthetic_data.py` | Generates a synthetic dataset with the same schema and class structure. |
| `Anxiety_Synthetic.xlsx` | A ready-to-use synthetic dataset (14 ordinal items, five severity classes). |
| `Results/` | Precomputed result tables (CSV) and figures from a full run. |

## Data availability

The original questionnaire responses contain sensitive student mental-health
data and identifiers and are therefore **not** shared. A fully synthetic dataset
(`Anxiety_Synthetic.xlsx`) with the same structure and a comparable class
distribution is provided so that the entire pipeline can be run and every
reported table reproduced without any real participant data.

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `tensorflow`,
  `xgboost`, `lightgbm`, `catboost`, `openpyxl`, `matplotlib`

Install:

```bash
pip install numpy pandas scikit-learn imbalanced-learn tensorflow xgboost lightgbm catboost openpyxl matplotlib
```

## How to run

### Option 1 — Google Colab (recommended)

1. Upload `HARS_pipeline_release.ipynb` to Colab.
2. Run the first cell to install the dependencies.
3. Set the data path in the configuration cell:
   - to use the synthetic data, upload `Anxiety_Synthetic.xlsx` and set
     `DATA_PATH = "Anxiety_Synthetic.xlsx"`;
   - to use your own data, point `DATA_PATH` to your file (the item columns must
     contain the marker text `pilih skor 0 - 4`, and an identifier column such as
     `NPM` is used for duplicate screening).
4. Select **Runtime → Run all**. All tables and figures are produced in order.

### Option 2 — Local

```bash
python generate_synthetic_data.py     # creates Anxiety_Synthetic.xlsx
jupyter notebook HARS_pipeline_release.ipynb
```

Set `DATA_PATH = "Anxiety_Synthetic.xlsx"` in the configuration cell, then run
all cells.

## Reproducibility

- A fixed random seed (`RANDOM_STATE = 42`) is set for NumPy, Python, and
  TensorFlow, and deterministic TensorFlow operations are enabled, so a
  top-to-bottom run reproduces identical results.
- The train/test split, five-fold cross-validation, SMOTE (applied to training
  folds only), and all model hyperparameters are defined explicitly in the
  notebook.
- Cross-validated results are reported as mean ± standard deviation with 95%
  confidence intervals; the proposed model is additionally evaluated over five
  independent stratified splits.

## Outputs

Running the notebook writes the following files to the output directory:
`main_performance_table.csv`, `per_class_recall_table.csv`,
`confusion_matrix_counts.csv`, `feature_ablation_table.csv`,
`missing_item_robustness_table.csv`, `noisy_response_robustness_table.csv`,
`repeated_cv_summary.csv`, `proposed_multiseed_summary.csv`,
`rule_vs_model_degradation.csv`, `reliability_analysis.csv`,
`class_distribution.csv`, and the corresponding figures.

## Citation

If you use this code, please cite the associated article (details to be added
upon publication).

## License

Released under the MIT License.
