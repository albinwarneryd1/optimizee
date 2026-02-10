Overview

Optimizee is a modular machine-learning application designed to demonstrate a complete workflow for electricity consumption analysis.
The project focuses on system structure, reproducibility, and clear separation of concerns rather than aggressive model optimization.

The application is built as a runnable system rather than a collection of notebooks, making it suitable as a foundation for further development or real-world data integration.

How it works

The application follows a simple but robust pipeline:

Raw data ingestion
CSV files containing time-series electricity consumption data are placed in data/raw/.

Preprocessing
The raw data is cleaned and transformed into a structured, model-ready dataset.
Time-based features such as hour of day, weekday, lagged values, and rolling statistics are generated.

Model training
A baseline regression model is trained on the processed data and evaluated using standard error metrics.

Visualization
An interactive dashboard visualizes historical consumption, model predictions, and basic insights derived from the data.

Each step can be executed independently through the command-line interface.

Command-line usage

Optimizee exposes a small CLI to ensure reproducible execution:

optimizee status
optimizee preprocess
optimizee train
optimizee dash


status checks whether processed data and a trained model exist

preprocess generates the processed dataset

train trains and saves the machine-learning model

dash launches the interactive dashboard

Model choice

The current implementation uses a RandomForest regression model as a baseline.
The goal is not to achieve maximum predictive performance, but to provide a stable and interpretable starting point that can easily be extended or replaced.

Data note

The project is currently demonstrated using synthetic time-series data that mimics realistic electricity consumption patterns.
This allows the full pipeline and application flow to be tested without relying on proprietary or sensitive datasets.

Extensibility

The project is structured to support future extensions, such as:

Integration of real household or grid-level electricity data

Cost-based optimization using electricity spot prices

Consumption forecasting

Automated reporting and export of results

Purpose

Optimizee is intended as a learning and portfolio project that emphasizes:

Clean project structure

Reproducible machine-learning workflows

Practical application design over isolated experimentation
