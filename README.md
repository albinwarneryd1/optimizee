Optimizee

Optimizee is a small end-to-end machine learning application for electricity consumption analysis. It is built as a modular and reproducible system that covers the full workflow from raw data ingestion to model training and visualization. The project is intended to demonstrate system structure and data flow rather than optimized predictive performance, and is currently demonstrated using synthetic time-series data.

The application is run through a simple command-line interface. After placing one or more CSV files in the data/raw directory, the data is preprocessed, a baseline model is trained, and an interactive dashboard is launched.

To run the application, install it in a virtual environment using pip install -e ., then execute optimizee preprocess, optimizee train, and optimizee dash from the project root.
