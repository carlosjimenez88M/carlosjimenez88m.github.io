---
author: Carlos Daniel Jiménez
date: 2024-11-21
title: "Artifact Design and Pipeline in MLOps Part I"
description: Introduction to Artifacts designs
categories: ["MLOps"]
tags: ["mlflow", "artifacts", "pipeline", "mlproject", "argparse"]
series:
  - mlops
---
## Artifact Design and Pipeline in MLOps Part I

In MLOps, most of the work focuses on the **inference stage**, specifically the development of microservices. However, the broader picture goes beyond this—it includes aspects ranging from programming practices to resource utilization that need to be evaluated. This is where the role of a **Machine Learning DevOps Engineer** becomes crucial. In this post, I want to address this profile by approaching it from the perspective of designing a model.

Typically, in Data Science, a pipeline is presented as follows:

 

![image.png](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/images/image.png?raw=true)

To make this process reproducible, we need to discuss **artifacts**, which are essentially "contracts" defined by code. For example, if we have an ETL process, the artifact might be a new database. Artifacts help us maintain an organized, modular, and sequential process within pipelines. Moreover, artifacts simplify experiments and model retraining by allowing us to define training rules for scenarios such as new data versions or significant data drift.

Although this formalizes the programming practices that data scientists implement, it is, in fact, a good practice—similar to including unit tests for every function or class in our code.

![image.png](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/images/artifacts.png?raw=true)

The above diagram illustrates the flow of creating an artifact through contracts. Now, where do these contracts come from? This is where **MLflow**—specifically, `MLproject`—comes into play. An `MLproject` serves as a manifesto of promises made by code or components, based on predefined conditions. The advantage of using such a manifesto is that it ensures each artifact is **independent and reproducible**. Its basic structure includes:

- **name**: The name of the manifesto.
- **conda_env**: The Conda environment that allows the installation of required packages or libraries for the component's contract.
- **entry_points**: Defines the steps required to execute the pipeline.

So far, we’ve defined two components for developing an artifact: the `conda.yaml` and the `MLproject`. Now, let’s talk about the code itself, which brings us to **argparse** functions.

---

Previously, we worked with immutable code, often derived from notebooks or experimental scripts. However, as data versioning became more frequent, it became necessary to introduce flexibility in parameter management. This is where **argparse** gained importance. Let’s see an example using the Iris dataset:

```python
#====================#
# ---- Libraries ----#
#====================#

import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#=========================#
# ---- Main Function ---- #
#=========================#

def main(args):
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )

    model = LogisticRegression(max_iter=args.max_iter, 
                                random_state=args.random_seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Logistic Regression model on the Iris dataset.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing (default: 0.2)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum number of iterations for Logistic Regression (default: 200)")

    args = parser.parse_args()
    
    main(args)

```

---

### **Taxonomy of the Code**
1. A `main` function is defined.
2. This function accepts parameters that can be modified via the command line or a `config.yaml` file.
3. An **argument parser** (`parser`) is initialized to define the parameters and their conditions.
4. The `args = parser.parse_args()` command seals these contracts.
5. The function executes with the established conditions.

---

### **Manifesto for This Example**

```yaml
name: iris-classifier

conda_env: conda.yaml

entry_points:
  train:
    parameters:
      test_size: 
        type: float
        default: 0.3
        description: proportion split into general dataset
      random_seed: 
        type: int
        default: 42
      max_iter: 
        type: int
        default: 200
    command: >-
      python train_iris.py --test_size {test_size}\ 
                           --random_seed {random_seed}\ 
                           --max_iter {max_iter}
```

This manifesto defines how the parameters are passed into the parsed code and their default values. To execute this component, you would run:

```bash
python train_iris.py --test_size 0.3 --max_iter 300
```

---

### **Orchestration**

The idea of a pipeline is not to execute these commands manually. Instead, a single command (e.g., `mlflow run .`) should suffice. To achieve this, the process must be orchestrated. Let’s consider an example pipeline with a GitOps component (kept simple for now). Imagine downloading a dataset. We’ll have a folder containing three files: `MLproject`, `conda.yaml`, and `main.py`.

#### **Main Script**
The `main.py` file downloads and registers the dataset as an artifact.

```python
#!usr/bin/env python
'''
End-to-End Machine Learning Project
Step: Download Dataset
2024-11-21
'''

#=====================#
# ---- Libraries ---- #
#=====================#

import os
import argparse
import logging
import pathlib
import pandas as pd
import tarfile
import urllib.request
import wandb

#=====================#
# ---- Functions ---- #
#=====================#

def load_housing_data(args):
    """
    Downloads the housing dataset, extracts it, registers it with W&B,
    and returns the loaded dataset as a pandas DataFrame.
    """
    # Logic for downloading and processing data
    ...

if __name__ == "__main__":
    # Argument parsing logic
    ...
```

#### **Manifesto**
```yaml
name: download_data
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      file_url:
        description: URL of the file to download
        type: str
      artifact_name:
        description: Name of the artifact
        type: str
      artifact_type:
        description: Data to train the model
        type: str
        default: raw_data
      artifact_description:
        description: Artifact to train the model
        type: str
    command: >-
      python main.py --file_url {file_url}\
                     --artifact_name {artifact_name}\
                     --artifact_type {artifact_type}\
                     --artifact_description {artifact_description}
```

---

By running `mlflow run .`, the dataset is downloaded and registered as the first artifact of the project. In the next post, I’ll discuss how to expand the pipeline, incorporating agents to find the best model parameters.