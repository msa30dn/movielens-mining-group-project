# MovieLens Mining Group Project

This repository contains part of a group project focused on applying **data mining techniques** to the [MovieLens dataset](https://grouplens.org/datasets/movielens/) for recommendation analysis.

🌐 Project Info.: [movielens-data-mining.web.app](https://movielens-data-mining.web.app/)

## Repository purpose

This repo provides the data-engineering and mining assets for **Story B.1 (basket-rule mining)** in the broader MovieLens project.

At a high level, it provides:
- A reusable preprocessing backbone that transforms raw MovieLens data into validated analytical tables and basket-style transaction views.
- A Story B.1 notebook that mines frequent itemsets and association rules, then produces interpretation outputs and analysis artifacts.
- Supporting scripts and dependency definitions used by the preprocessing and mining workflow.

## What this repository includes

- `preprocessing/core/`: the core pipeline package that handles ingestion, cleaning, validation, temporal splitting, feature generation, transaction assembly, reduction, and auditing.
- `preprocessing/scripts/`: orchestration entrypoints for staging raw inputs and producing processed run outputs.
- `story-module-b1/B1_Story_Module.ipynb`: the Story B.1 analysis notebook for basket mining, rule generation, quality filtering, decoding, and diagnostics.
- `story-module-b1/story_b1_demo_app/`: a lightweight demo application for Story B.1 outputs.
- `requirements.txt`: Python dependency set for the project modules and notebook workflows.

## What it provides (outputs)

The preprocessing and Story B.1 components together produce:
- **Structured data artifacts** such as clean dimensions/facts, train-test interaction tables, features, and transaction tables.
- **Reduced basket artifacts** for mining-focused experiments (including token and basket statistics).
- **Association-rule artifacts** and related diagnostics used to evaluate recommendation patterns and cross-genre behavior.
- **Run-level reports** (metadata, audit summaries, and analysis outputs) that support reproducibility and interpretation.

## Story B.1 scope in this repo

Story B.1 in this repository is focused on **association-rule discovery from user behavior baskets**. The notebook layer consumes preprocessing outputs and supports:
- Frequent itemset mining (pair-focused for interpretability)
- Association-rule generation with threshold-based filtering
- Rule quality gating and popularity-bias-aware diagnostics
- Decoded, human-readable rule exploration and summary analysis

## Team & collaboration

This is a collaborative group project. Contributions are tracked through Git commits and pull requests.

## License

Licensed under the [Apache License 2.0](LICENSE).
