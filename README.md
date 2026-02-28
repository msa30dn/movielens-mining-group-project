# MovieLens Mining Group Project

This repository contains part of a group project focused on applying **data mining techniques** to the [MovieLens dataset](https://grouplens.org/datasets/movielens/) for recommendation analysis.

🌐 Project Info.: [movielens-data-mining.web.app](https://movielens-data-mining.web.app/)

## What is in this repository

The current repository snapshot includes:

- `story-module-b1/B1_Story_Module.ipynb`: an evaluation and interpretation notebook for Story B.1 basket-rule mining outputs.
- `requirements.txt`: Python dependencies for data processing, mining, visualization, and notebook/app workflows.

## Story B.1 notebook overview

`B1_Story_Module.ipynb` is a **read-only evaluation layer** designed to analyze outputs produced by a separate mining notebook (`B_basket_rules.ipynb`).

It focuses on:

- Association rule quality diagnostics (support, confidence, lift)
- Popularity-bias analysis using token statistics
- Rule coverage and structural summaries
- Human-readable interpretation helpers for rule neighborhoods
- Optional held-out recommendation metrics (e.g., hit-rate@K, MRR) for movie-only rule runs

The notebook expects previously generated artifacts under Story B output folders (including parquet tables and run reports), then summarizes and interprets the mined rule sets.

## Quick start

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook story-module-b1/B1_Story_Module.ipynb
```

> Note: The notebook evaluates existing mining outputs; ensure the expected Story B artifact directories are available in your local data layout.

## Team & collaboration

This is a collaborative group project. Contributions are tracked through Git commits and pull requests.

## License

Licensed under the [Apache License 2.0](LICENSE).
