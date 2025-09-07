# Machine Learning for Higgs Boson Detection

## Overview

[cite_start]This project investigates b-quark tagging for the detection of the Higgs boson, which is a decay product of Higgs interactions[cite: 6, 7]. A traditional cut-based analysis is performed to establish a baseline, which is then compared against a machine learning approach using a neural network. [cite_start]The objective is to develop a more sensitive tagging algorithm to distinguish Higgs boson signal events from significant background noise[cite: 27].

## Key Results

The primary metric for performance is signal sensitivity. The two methods yielded the following results:

-   [cite_start]**Cut-Based Analysis Sensitivity**: **1.863** [cite: 8]
-   [cite_start]**Machine Learning Model Sensitivity**: **$2.767_{-0.014}^{+0.026}$** [cite: 9]

[cite_start]The machine learning model demonstrates a **48.5% improvement** in sensitivity over the traditional cut-based method[cite: 517].

## Repository Structure

-   `Higgs-Cut-based selection (1).ipynb`: A Jupyter Notebook containing the code for the cut-based analysis, which serves as a performance baseline.
-   `Higgs-Machine Learning (1).ipynb`: A Jupyter Notebook that details the data pre-processing, model architecture, training, and evaluation of the neural network.
-   [cite_start]`b4 cut.png` / `after cut.png`: Plots showing the reconstructed b-quark mass ($m_{BB}$) distribution before and after the cuts were applied in the baseline analysis[cite: 55, 56].
-   `requirements.txt`: A list of the Python dependencies required to run the notebooks.

## Methodology

### 1. Data Pre-processing

Before training the model, the data undergoes several pre-processing steps:
-   [cite_start]**Data Cleaning**: Rows with missing values are removed[cite: 116].
-   [cite_start]**Normalisation**: All features are scaled to have a mean of zero and a standard deviation of one to ensure equitable influence on the model[cite: 120].
-   [cite_start]**Feature Removal**: The `nJ` and `nTags` features were removed as they held constant values of two and thus provided no discriminative information[cite: 122, 123].

### 2. Cut-Based Analysis

[cite_start]A baseline sensitivity was established by applying sequential cuts on four key kinematic variables: `Mtop`, `dRBB`, `pTB2`, and `pTV`[cite: 46, 48]. [cite_start]This method provides a benchmark against which the machine learning model's performance is assessed[cite: 44].

### 3. Machine Learning Model

[cite_start]A fully connected feed-forward neural network was designed and tuned for this binary classification task[cite: 125].
-   [cite_start]**Architecture**: The model consists of an input layer, four hidden layers, and a single-unit output layer[cite: 176].
-   [cite_start]**Activation Functions**: The ReLU activation function is used for all hidden layers [cite: 220][cite_start], while a Sigmoid function is used for the output layer to produce a probability score[cite: 216, 217].
-   [cite_start]**Optimiser and Loss Function**: The model is compiled using the Nadam optimiser and the BinaryFocalCrossentropy loss function, a combination which was found to yield consistently high sensitivity[cite: 185].
-   [cite_start]**Training Parameters**: The model was trained with a learning rate of 0.005, a batch size of 128, and for 10 epochs[cite: 292].

## Getting Started

To reproduce the results in this repository, please follow the steps below.

### Prerequisites

Ensure you have Python 3 installed. This project requires the Python libraries listed in the `requirements.txt` file.

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/daydreamerovo/Machine-Learning-in-Higgs-Boson-Detection.git](https://github.com/daydreamerovo/Machine-Learning-in-Higgs-Boson-Detection.git)
    cd Machine-Learning-in-Higgs-Boson-Detection
    ```

2.  Install the required dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  To run the baseline analysis, launch Jupyter Notebook and open `Higgs-Cutbased selection (1).ipynb`.
    ```bash
    jupyter notebook "Higgs-Cutbased selection (1).ipynb"
    ```
2.  To train and evaluate the machine learning model, open and run the cells in `Higgs-Machine Learning (1).ipynb`.
    ```bash
    jupyter notebook "Higgs-Machine Learning (1).ipynb"
    ```
