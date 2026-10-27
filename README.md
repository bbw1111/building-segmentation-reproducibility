# Building Segmentation Reproducibility

## Overview

This repository contains experimental notebooks and selected result files associated with a research study on remote-sensing building segmentation.

The study investigates building segmentation using different model configurations and edge-related supervision strategies. The repository is intended to document the experimental settings and support inspection of the reported results.

## Repository Contents

### Experimental Notebooks

- `whu_seed42_experiments.ipynb`: WHU building segmentation experiments associated with seed 42.
- `whu_seed2026_and_baselines.ipynb`: Additional WHU experiments and baseline comparisons.
- `edge_supervision_ablation.ipynb`: Experiments examining edge-supervision settings.
- `inria_legacy_experiments.ipynb`: Earlier Inria experiments retained for reference. These experiments should not automatically be interpreted as the final evaluation protocol used in the manuscript.

### Results

The `结果/` directory contains selected experimental result files:

- `independent_test/X2_shallow_test_mean_SD.csv`: Summary statistics for X2-shallow independent-test results.
- `validation/X2_shallow_validation_mean_SD.csv`: Summary statistics for X2-shallow validation results.
- `validation_final/X2_region_PIL_3seed_validation_mean_SD.csv`: Three-seed validation summary for X2-region.
- `规范结果.csv`: A consolidated record of selected experimental values and their provenance.

The reported values should be interpreted according to the evaluation protocol and experiment descriptions in the associated manuscript.

## Datasets

The experiments use the WHU building dataset and the Inria Aerial Image Labeling Dataset.

The original datasets are not included in this repository. Users should obtain them from their official sources and comply with the applicable access conditions and usage licenses.

Dataset-specific preprocessing and directory settings may need to be adapted to the local environment.

## Environment and Dependencies

The `requirements.txt` file provides the currently recorded Python dependencies.

Some notebooks were developed in cloud-based environments. Package versions, hardware availability, dataset paths, and notebook settings may need to be checked before execution.

The current repository should not be assumed to reproduce every reported experiment without additional environment and artifact verification.

## Reproducibility Notes

The repository is being organized from an existing experimental archive. The notebooks and result files serve as supporting research materials.

Before attempting to reproduce a result, verify the corresponding model configuration, random seed, dataset split, checkpoint, and evaluation protocol.

Historical or exploratory experiments should be distinguished from the formal experiments reported in the manuscript.

## Data and Code Availability

The availability of code and supporting experimental materials is subject to the applicable ownership, licensing, and redistribution conditions.

A permanent repository link and a definitive description of the publicly available materials should be provided in the manuscript only after the repository contents and sharing permissions have been verified.
