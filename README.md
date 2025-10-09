[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.17303249-blue)](https://doi.org/10.5281/zenodo.17303249)
[![DOI](https://img.shields.io/badge/DOI-10.1101/2025.09.21.25336258-blue)](https://doi.org/10.1101/2025.09.21.25336258)

Biofluid Colorimetric Sensors
==============================

Read signals from colorimetric sensors even in non-uniform illumination.

Details on Models and Data can be found at https://www.medrxiv.org/content/10.1101/2025.09.21.25336258v1

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── main.py        <- Main script to run the pipeline
    │   │
    │   ├── data          <- Scripts to download or generate data
    │   │   
    │   ├── train          <- Scripts to train the denoising/prediction models
    │   │
    │   ├── test          <- Scripts to evaluate the denoising/prediction models
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    └── 

------
