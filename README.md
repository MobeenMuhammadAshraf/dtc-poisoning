# Hidden Recurrences: Robustness of Thyroid Cancer Recurrence Predictors to Targeted Training-Label Poisoning

Code and data for the paper:

> Author names. Hidden Recurrences: Robustness of Thyroid Cancer Recurrence Predictors to Targeted Training-Label Poisoning. JMIR Medical Informatics (submitted).

## Dataset

The UCI Differentiated Thyroid Cancer Recurrence dataset (Borzooei and Tarokhian, 2023):
- Source: https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence
- Original: 383 patients, 17 columns
- Target: Recurred (Yes/No), 108 recurrences
- Response excluded as a post-treatment variable
- Duplicates removed: 19 rows (all non-recurrences), leaving 364 patients and 108 recurrences

## Repository structure

    .
    |-- data/
    |   |-- Thyroid_Diff.csv
    |   |-- X_prepared_dedup.csv
    |   `-- y_prepared_dedup.csv
    |-- code/
    |   |-- 01_prepare_data.py
    |   |-- 02_run_experiments.py
    |   `-- 03_analysis.py
    |-- results/
    |   |-- T1_clean_baseline.csv
    |   |-- T2_poisoning.csv
    |   |-- T3_defences.csv
    |   |-- T3_summary.csv
    |   |-- T3_wilcoxon_fdr.csv
    |   |-- T4_threshold.csv
    |   `-- full_results_dedup.csv
    |-- requirements.txt
    `-- README.md

## Reproducing the paper

    pip install -r requirements.txt
    python code/01_prepare_data.py
    python code/02_run_experiments.py
    python code/03_analysis.py

Seeds: 42, 7, 2024. Metrics are means across 15 runs (3 seeds times 5 stratified folds).

## License

MIT