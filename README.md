# MovieLens Mining Group Project

This repository contains part of a group project focused on applying **data mining techniques** to the [MovieLens dataset](https://grouplens.org/datasets/movielens/) for recommendation analysis.

🌐 Project Info.: [movielens-data-mining.web.app](https://movielens-data-mining.web.app/)

## What is in this repository

The current repository snapshot includes:

- `preprocessing/`: core data pipeline that ingests raw MovieLens CSV files, builds clean train/test tables, engineers features, and prepares Story B.1 transaction artifacts.
- `story-module-b1/B1_Story_Module.ipynb`: Story B.1 mining notebook that runs frequent itemset and association-rule analysis over prepared transactions.
- `requirements.txt`: Python dependencies for preprocessing, mining, visualization, and notebook/app workflows.

## Preprocessing (core pipeline)

The `preprocessing/core` package is the project backbone. It handles staged ingestion, validation, table contracts, split generation, feature creation, transaction construction, reduction, and audit reporting.

### Pipeline phases

- **Phase 0-1 (ingest/staging):** Read raw CSVs and write Parquet staging files (`movies_raw.parquet`, `ratings_raw.parquet`, `tags_raw.parquet`, `links_raw.parquet`, plus optional genome files).
- **Phase 2 (clean + validate):** Build canonical clean tables and run schema/data validation.
- **Phase 3 (temporal split):** Build train/test interaction splits (`global` or `per_user` policy).
- **Phase 4 (feature factory + transactions):** Generate user/movie features and `transactions_train.parquet`.
- **Phase 4.5 (reduction):** Create `transactions_train_reduced.parquet` plus token/basket reduction diagnostics.
- **Phase 5 (audit):** Write end-of-run audit checks and summary gates.

### Run preprocessing

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Build staging files once from the raw MovieLens folder:

```bash
python preprocessing/scripts/build_raw_staging.py --raw data/raw/ml-latest --staging data/raw_staging
```

3. Run the processed pipeline:

```bash
python preprocessing/scripts/build_processed.py --staging data/raw_staging --out data/processed
```

Key CLI options for processed runs:
- `--split-policy {global,per_user}`
- `--test-frac <float>`
- `--cutoff-date <YYYY-MM-DD[ HH:MM:SS]>` (global policy)
- `--run-tag <name>`

Pipeline outputs are versioned under `data/processed/<run_id>/` with:
- `tables/` (clean tables, train/test interactions, features, transactions, reduced transactions)
- `reports/` (build logs, metadata, split/reduction reports, and audit report)

## Story B.1 notebook overview

`story-module-b1/B1_Story_Module.ipynb` is the Story B.1 mining and analysis layer. It consumes preprocessing outputs (especially transaction tables and lookups) and focuses on:

- Basket preparation for `mlxtend`
- Frequent itemset mining (pair-focused)
- Association-rule generation with confidence/lift thresholds
- Optional rule quality gates for stronger evidence and anti-popularity controls
- Rule decoding/categorization and exploratory visualizations
- Cross-genre structure analysis and output artifact writing

Open the notebook with:

```bash
jupyter notebook story-module-b1/B1_Story_Module.ipynb
```

> Note: Run preprocessing first so Story B.1 has the expected transaction and lookup inputs.

## Team & collaboration

This is a collaborative group project. Contributions are tracked through Git commits and pull requests.

## License

Licensed under the [Apache License 2.0](LICENSE).
