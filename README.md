# PINNs for Traffic State Estimation

A **physics-informed deep learning framework** for traffic state estimation, leveraging macroscopic traffic flow models (**LWR** and **ARZ**) to reconstruct traffic density and velocity from sparse sensor data.

This project implements and evaluates **Physics-Informed Neural Networks (PINNs)** for reconstructing continuous traffic state fields (density and velocity) using the real-world **Next Generation Simulation (NGSIM)** dataset. This is a research paper implementation. To learn more, visit [this website](https://ieeexplore.ieee.org/document/9531557).

---

## Project Overview

This system addresses the challenge of **Traffic State Estimation (TSE)** by integrating fundamental traffic flow physics directly into a deep learning model.  
By enforcing the **Lighthill-Whitham-Richards (LWR)** and **Aw-Rascle-Zhang (ARZ)** conservation laws as soft constraints, the model generates **physically consistent and accurate** estimations of traffic dynamics, even with sparse data.

The project provides a complete workflow, from fetching and processing raw NGSIM vehicle trajectory data to training and evaluating PINNs against several baseline models.

---

## Key Capabilities

- **Physics-Informed Traffic State Estimation** – Implements PINNs for both first-order (LWR) and second-order (ARZ) traffic flow models.  
- **Fundamental Diagram Learning** – Includes a `FDLearner` module to learn the complex relationship between traffic density and velocity directly from data.  
- **Comprehensive Model Comparison** – Benchmarks the PINN approach against baseline models: LSTM, Vanilla MLP, and Linear Regression.  
- **Real-World Data Application** – Validates models using the NGSIM dataset in a realistic highway scenario.  
- **End-to-End Data Pipeline** – Includes scripts to fetch, clean, and aggregate traffic data for training.

---

## Directory Structure

```
PINNs-for-Traffic-State-Estimation/
│
├── Dataset/
│   └── Note.md                                # Details about the dataset used
│
├── Notebooks/
│   ├── DataCleaning.ipynb                     # Fetches, cleans, and grids raw NGSIM data
│   └── Exploration.ipynb                      # Trains and evaluates all models (PINN, LSTM, etc.)
│
├── Scripts/
│   ├── lstm.py                                # Defines the LSTM model architecture
│   ├── physics.py                             # Defines PDE residuals for physics loss calculation
│   ├── pinn_model.py                          # Defines the PINN and FDLearner architectures
│   └── vanilla.py                             # Defines the Vanilla MLP architecture
│
└── README.md                                  # Project documentation
```

---

## Workflow Description

### 1. Data Processing & Gridding  
**File:** `Notebooks/DataCleaning.ipynb`

- Fetches raw vehicle trajectory data from the U.S. Department of Transportation’s public API.  
- Converts units (e.g., mph → m/s) and standardizes column names.  
- Aggregates data onto a discrete spatio-temporal grid $$\((x, t)\)$$ to compute density $$\(ho\)$$ and velocity $$\(u\)$$.  
- Normalizes the grid and saves the final data as `.csv` files.

---

### 2. Model Development & Training  
**Files:** `Scripts/`, `Notebooks/Exploration.ipynb`

- `pinn_model.py` – Defines the core PINN and FDLearner architectures.  
- `physics.py` – Implements ARZ PDEs and computes physics residuals via `torch.autograd`.  
- `lstm.py` & `vanilla.py` – Baseline architectures.  
- The `Exploration.ipynb` notebook loads the processed data and trains all models (PINN-ARZ, MLP, LSTM, Linear Regression).

---

### 3. Model Evaluation & Comparison  
**File:** `Notebooks/Exploration.ipynb`

- Models are tested on a held-out validation dataset from a different time window.  
- Evaluation metrics: **MSE** and **MAE** for both density and velocity.  
- The **physics residual** is calculated for the PINN model to quantify physical consistency.

---

## 📊 Data Management

### Primary Dataset
**NGSIM (Next Generation Simulation) Dataset** – Detailed real-world vehicle trajectories from “Peachtree Street” and “US-101”.

- Metadata: `Dataset/1-_US_101_Metadata_Documentation.pdf`

### Processed Data
Files like:
```
peachtree_dir2_lane1_10m_1s_10min.csv
```
contain normalized gridded data ($$\(x_{nd}, t_{nd}, rho_{nd}, u_{nd}, mask)\$$) for model training and evaluation.

---

## Technical Architecture

| **Property** | **Details** |
|---------------|-------------|
| **Architecture** | Fully-connected feedforward (8 layers × 20 neurons) |
| **Input Features** | $$\(x\)$$ (normalized space), $$\(t\)$$ (normalized time) |
| **Output Variables** | $$\({rho\})$$ density, $$\(u\)$$ velocity |
| **Activations** | Tanh |
| **Optimizer** | Adam |
| **Physics Law** | Aw-Rascle-Zhang (ARZ) second-order model |
| **Reference Paper** | *A Physics-Informed Deep Learning Paradigm for Traffic State and Fundamental Diagram Estimation* |

---

## Prerequisites

### System Requirements
- Python 3.10+
- Jupyter Notebook / JupyterLab

### Required Libraries
```bash
pip install torch pandas numpy requests scikit-learn matplotlib
```

---

## 🛠️ Installation Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/AlphaParticle28/Traffic-Flow-PINNs
cd PINNs-for-Traffic-State-Estimation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
The data is fetched automatically via API.  
Ensure you have an active internet connection when running the data cleaning notebook.

---

## How to Run

### Step 1 – Fetch and Process Data
```bash
jupyter notebook Notebooks/DataCleaning.ipynb
```

### Step 2 – Train and Evaluate Models
```bash
jupyter notebook Notebooks/Exploration.ipynb
```

This will:
- Load the processed training data  
- Train the PINN-ARZ and baseline models  
- Evaluate on validation data  
- Print performance metrics

---

## Key Features

| **Feature** | **Description** | **Benefit** |
|--------------|-----------------|--------------|
| **Physics-Informed Loss** | Embeds ARZ PDEs into the loss function | Ensures physically realistic predictions |
| **Baseline Comparison** | Compares PINN vs. MLP, LSTM, Linear Regression | Shows trade-offs between accuracy and consistency |
| **Automatic Differentiation** | Uses `torch.autograd` for PDE derivatives | Simplifies and accelerates physics loss computation |
| **End-to-End Workflow** | From data fetching → training → evaluation | Fully reproducible real-world application |

---

## Model Performance

Validation Set Results:

| Model | Total MSE (ρ + u) | MSE (ρ) | MSE (u) | Physics Residual |
|--------|------------------:|--------:|--------:|------------------:|
| **PINN-ARZ** | 0.5580 | 0.0808 | 0.4773 | 7.6e-5 |
| **LSTM** | 0.1953 | 0.0751 | 0.1202 | N/A |
| **Vanilla MLP** | 0.2153 | 0.0638 | 0.1515 | N/A |
| **Linear Regression** | 0.2285 | 0.0642 | 0.1643 | N/A |

The **extremely low physics residual** of the PINN-ARZ model indicates strong adherence to the governing laws of traffic flow — leading to better generalization and more trustworthy predictions in data-sparse regions.

---

## Technical Implementation

### PINN Composite Loss Function

The model minimizes:
$$\ L_{\text{total}} = L_{\text{data}} + \lambda_{\text{physics}} \cdot L_{\text{physics}}\ $$

Where:
- $$\ L_{\text{data}}\ $$: Mean Squared Error between predicted and observed data  
- $$\ L_{\text{physics}}\ $$: Mean Squared Error of PDE residuals (physics consistency)

---
### Lighthill-Whitham-Richards (LWR) Model

**Physics Loss Function (Diffusion Equation):**

$$\ f := \frac{\partial \rho}{\partial t} + \frac{\partial Q(\rho)}{\partial x} - \epsilon \frac{\partial^2 \rho}{\partial x^2}$$

### Aw-Rascle-Zhang (ARZ) Model Equations

**Physics Loss Function 1 (Conservation of Mass):**

$$\ f_1 := \frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} \$$

**Phsyics Loss Function 2 (Momentum-like Equation):**

$$\ f_2 := \frac{\partial (u + h(\rho))}{\partial t} + u \frac{\partial (u + h(\rho))}{\partial x} - \frac{U_{eq}(\rho) - u}{\tau} \$$

Where:
- $$\ \rho(x,t)\ $$: Traffic density
- $$\ Q(\rho\) $$: Traffic flux  
- $$\ u(x,t)\ $$: Traffic velocity
- $$\ \epsilon\ $$: Diffusion constant
- $$\ U_{eq}(\rho)\ $$: Equilibrium speed (learned by FDLearner)  
- $$\ h(\rho)\ $$: Traffic pressure term  
- $$\ \tau\ $$: Relaxation time constant

---
