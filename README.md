# Home Credit Default Risk

Predicting loan default for applicants with little or no credit history.

**Test ROC-AUC: 0.7764** — reproducible from raw data with `python -m src.train`.

## The problem

Home Credit lends to people excluded from mainstream banking — applicants
with no credit file, who a traditional scorecard would reject automatically.
Serving them is the reason the business exists.

The data shows why that's hard. Applicants with no credit bureau history
default at **10.1%** versus **7.7%** for those with a history. The population
the company was founded to serve is measurably riskier, so any model that
scores well will learn to reject them. The model works against the mission
unless someone intervenes deliberately at the threshold.

## Data

307,511 applications, 8.07% default rate, 122 columns in the main table plus
six supporting tables. 67 columns have missing values; 41 are over half empty.

## What each data source was actually worth

| Features | CV AUC | Gain |
|---|---:|---:|
| Application table only | 0.7536 | — |
| + 5 engineered ratios | 0.7601 | +0.0065 |
| + bureau (1.7M rows → 17 cols) | 0.7647 | +0.0046 |
| + previous applications (1.67M rows → 18 cols) | 0.7686 | +0.0039 |

**Five hand-built ratio features contributed more than either million-row
table.** `CREDIT_TERM` (annuity ÷ credit) ranks 5th in model importance —
above the raw amounts it's built from, and above age and employment length.

## Which feature engineering worked, and why

Four groups were tested independently against the baseline:

| Group | CV AUC | Verdict |
|---|---:|---|
| Ratios | 0.7601 | +0.0065 — kept |
| Time transforms | 0.7534 | no effect — dropped |
| EXT_SOURCE combinations | 0.7533 | no effect — dropped |
| Document/contact flags | 0.7536 | no effect — dropped |
| All combined | 0.7602 | +0.0001 over ratios alone |

Seventeen features were dropped for a cost of 0.0001 AUC.

The pattern: **gradient boosting benefits from features it cannot construct
itself.** A ratio requires division, which a tree cannot express by splitting
on two columns separately. `AGE_YEARS` is `DAYS_BIRTH` times a constant, and
`EXT_MEAN` restates columns the model already has — trees build their own
interactions, so both added nothing.

## Business metric

AUC doesn't tell a lender what to do. This does:

| Reject riskiest | Defaults caught | Good customers rejected | Lift vs random |
|---:|---:|---:|---:|
| 5% | 20.0% | 3.7% | 4.0× |
| 10% | 32.6% | 8.0% | 3.3× |
| 20% | 51.5% | 17.2% | 2.6× |
| 30% | 64.4% | 27.0% | 2.1× |

Past 30%, each additional default caught costs increasingly many good
applicants — an argument for where the cutoff belongs.

## Error analysis

**The model is well calibrated overall.** Mean predicted 0.0806 against an
actual rate of 0.0807. Brier score 0.0673 versus 0.0742 for always predicting
the base rate.

**It is calibrated for thin-file applicants too** — predicted 10.12% against
an actual 10.11%. It is not over-estimating their risk.

**But a single global threshold still rejects them at 28.3% versus 18.6%
for everyone else.** The disparity comes from the population's genuine risk
distribution, not from model error. No amount of model improvement removes
it; only a threshold policy decision does.

**The model ranks less well within that group** — AUC 0.7515 versus 0.7713.
It gets the group average right while being less able to distinguish good
thin-file applicants from bad ones.

**Weakest segment: retirees.** Ages 60-70 score 0.7294, pensioners 0.7459,
and the `DAYS_EMPLOYED` placeholder group 0.7471 — largely the same people.
Employment-based features carry no information for them, and those features
do significant work elsewhere in the model.

**Confident false positives inherit external error.** The 151 good customers
flagged above 50% risk had external scores of 0.20, 0.13 and 0.15 against
roughly 0.6 for missed defaulters. The model trusts EXT_SOURCE heavily, so
it inherits whatever the external bureaus got wrong.

## Data quality findings

- `DAYS_EMPLOYED = 365243` appears for 44,143 applicants (17.9%). It's a
  placeholder, and 44,126 of them are pensioners — a population marker, not
  dirt. They default at 5.37% versus 8.66%.
- The same 365243 code recurs in three columns of `previous_application`.
- Missingness carries signal: `AMT_REQ_CREDIT_BUREAU_*` missing implies a
  10.34% default rate versus 7.72% when present.

## Tests

`pytest` covers the feature transformations. The infinity test caught a real
bug: `add_ratios` divided by `AMT_INCOME_TOTAL` without guarding against zero.
No training row has zero income, so the bug was invisible in the data — but
it would have failed on new inputs.

## Repository

```text
home-credit-default-risk/
├── data/
│   └── raw/                        # not tracked — downloaded via Kaggle API
├── models/
│   └── lgbm_final.joblib           # model + column order
├── notebooks/
│   ├── 01_first_look.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_bureau.ipynb
│   ├── 06_previous_application.ipynb
│   └── 07_error_analysis.ipynb
├── reports/
│   ├── figures/
│   ├── bureau_experiments.csv
│   ├── feature_experiments.csv
│   └── table_contributions.csv
├── src/
│   ├── data.py                     # loading and table aggregation
│   ├── features.py                 # row-wise feature engineering
│   ├── train.py                    # end-to-end training
│   └── evaluate.py                 # group AUC, capture rates, calibration
├── tests/
|   └──test_features.py
├── requirements.txt
└── README.md
```


## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
kaggle competitions download -c home-credit-default-risk -p data/raw
cd data/raw && unzip home-credit-default-risk.zip && cd ../..
python -m src.train
```

## Limitations

- Four of seven tables unused. Based on the diminishing returns above,
  expect roughly +0.005-0.010 from all four.
- Hyperparameters untuned — LightGBM defaults with 300 trees.
- Trained on 3.8 GB RAM, which constrained cross-validation to 3 folds.