# Building Segmentation Reproducibility

## 1. Overview

This repository provides selected experimental notebooks, dataset split information, environment documentation, and result summaries associated with a study on remote-sensing building segmentation.

The study investigates the effects of decoder design, edge-related supervision, and feature reuse on building segmentation performance.

The repository is intended to support the inspection of experimental settings and reported results.

## 2. Repository Structure

### 2.1 Experimental Notebooks

- `whu_seed42_experiments.ipynb`: WHU building segmentation experiments associated with random seed 42.
- `whu_seed2026_and_baselines.ipynb`: Additional WHU experiments and baseline comparisons.
- `edge_supervision_ablation.ipynb`: Experiments examining different edge-supervision settings.
- `inria_legacy_experiments.ipynb`: Earlier Inria experiments retained as historical experimental records. These experiments should not be interpreted as the final evaluation protocol reported in the manuscript.

### 2.2 Dataset Split Information

The `inria_split/` directory contains supporting materials for the Inria tile-level split protocol:

- `README_Inria_Split_Protocol.md`: Description of the split protocol.
- `split_protocol.txt`: Summary of the dataset partitioning procedure.
- `tile_split_master.txt`: Source-tile split information.
- `tile_patch_mapping_generator.py`: Script for regenerating the tile-to-patch mapping.

These files document the source-tile separation used to reduce potential spatial data leakage between training and validation subsets.

### 2.3 Experimental Results

The `result/` directory contains selected experimental result summaries:

- `independent_test/X2_shallow_test_mean_SD.csv`: Summary statistics for X2-shallow independent-test results.
- `validation/X2_shallow_validation_mean_SD.csv`: Summary statistics for X2-shallow validation results.
- `validation_final/X2_region_PIL_3seed_validation_mean_SD.csv`: Three-seed validation summary for X2-region.
- `WHU_9CKPT_mean_SD.csv`: WHU experimental summary.

The root-level file `CANONICAL_RESULTS.csv` provides a consolidated record of selected experimental values.

The reported results should be interpreted together with the corresponding experimental configurations and evaluation protocols described in the manuscript.

## 3. Datasets

The experiments use the WHU Building Dataset and the Inria Aerial Image Labeling Dataset.

The original datasets are not redistributed in this repository. Users should obtain the datasets from their respective official sources and comply with the applicable access conditions and licenses.

Dataset directory paths may need to be adjusted according to the local environment.

## 4. Environment and Dependencies

The repository includes:

- `requirements.txt`: Recorded Python package dependencies.
- `ENVIRONMENT.md`: Additional environment and execution information.

Some experiments were developed in cloud-based computing environments. Package versions, hardware availability, dataset paths, and notebook configurations may require adaptation before execution.

## 5. Reproducibility Notes

The repository provides selected supporting materials from the experimental archive rather than a fully self-contained reproduction package.

Before reproducing or interpreting a result, users should verify the corresponding model configuration, random seed, dataset split, checkpoint availability, and evaluation protocol.

Historical and exploratory experiments are distinguished from the formal experimental settings described in the manuscript.

The absence of original datasets and model checkpoint files means that some experiments may require additional materials to be independently reproduced.

## 6. Data and Code Availability

This repository provides selected experimental notebooks, dataset split information, environment documentation, and result summaries for research inspection.

The original WHU and Inria datasets are not included and must be obtained from their respective sources. Model checkpoint files and other materials not explicitly listed in this repository are not claimed to be publicly available.

The repository should therefore be regarded as a collection of supporting research materials, rather than a guarantee of complete end-to-end reproducibility for every reported experiment.
