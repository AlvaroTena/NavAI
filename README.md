# NavAI 🛰️

## Artificial Intelligence for GNSS Optimization

NavAI is an advanced project focused on **Artificial Intelligence-Based Optimization of Precise Point Positioning (PPP)** to achieve **Adaptive Multipath Mitigation and Measurement Weighting** in GNSS Positioning systems. This project leverages state-of-the-art **Machine Learning** and **Reinforcement Learning** techniques to enhance accuracy and performance in challenging GNSS environments.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-blue.svg)](https://python-poetry.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-green.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://docker.com)

---

## 🎯 Key Features

- **🧠 MLNav Pipeline**: Machine Learning approach using unsupervised clustering for multipath detection and mitigation
- **🎮 RLNav Pipeline**: Reinforcement Learning approach using PPO (Proximal Policy Optimization) for adaptive positioning
- **🔧 Position Engine Integration**: Seamless wrapper for Position Engine (PE) integration
- **📊 Comprehensive Analytics**: Advanced reporting and visualization capabilities
- **🐳 Docker Support**: Containerized development environment
- **📈 Experiment Tracking**: Integration with Neptune for experiment monitoring
- **🔄 Data Versioning**: DVC-based data pipeline management

---

## 📚 Table of Contents

- [NavAI 🛰️](#navai-️)
  - [Artificial Intelligence for GNSS Optimization](#artificial-intelligence-for-gnss-optimization)
  - [🎯 Key Features](#-key-features)
  - [📚 Table of Contents](#-table-of-contents)
  - [🏗️ Project Architecture](#️-project-architecture)
    - [MLNav Pipeline](#mlnav-pipeline)
    - [RLNav Pipeline](#rlnav-pipeline)
    - [Shared Components](#shared-components)
  - [📁 Project Structure](#-project-structure)
  - [⚙️ Installation](#️-installation)
    - [Prerequisites](#prerequisites)
    - [Quick Setup](#quick-setup)
    - [Development Setup](#development-setup)
  - [🚀 Usage](#-usage)
    - [MLNav Pipeline](#mlnav-pipeline-1)
    - [RLNav Pipeline](#rlnav-pipeline-1)
    - [Configuration](#configuration)
  - [🐳 Docker Development](#-docker-development)
    - [Requirements](#requirements)
    - [Building Images](#building-images)
    - [Running Containers](#running-containers)
  - [🔧 Configuration](#-configuration)
    - [MLNav Configuration](#mlnav-configuration)
    - [RLNav Configuration](#rlnav-configuration)
  - [📊 Notebooks and Examples](#-notebooks-and-examples)
  - [🧪 Testing](#-testing)
  - [📈 Monitoring and Logging](#-monitoring-and-logging)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)

---

## 🏗️ Project Architecture

### MLNav Pipeline
**Machine Learning approach for multipath mitigation using clustering analysis:**

**Objective**: The MLNav pipeline applies lightweight machine learning to improve the GSHARP PPP algorithm by reducing multipath effects on GNSS measurements through clustering analysis of primary GNSS data points.

**Key Components:**
- **GNSS Data Clustering**: Uses K-means clustering to group GNSS measurements and discover patterns in multipath interference
- **Pattern Analysis**: After clustering, analysis of each group reveals what each cluster represents in terms of multipath characteristics
- **Real-world Training**: Trained on ~50 hours of data from diverse environments (highways, urban areas) ensuring robustness and applicability
- **Lightweight ML**: Designed for easy integration and as foundation for future advanced deep learning methods

**Technical Flow:**
1. **Data Input**: Reads offline processed GNSS datasets (text files with PE-computed features)
2. **Data Processing**: Concatenates multiple scenario files into unified datasets
3. **Feature Engineering**: Extract and standardize features from processed GNSS measurements
4. **Normalization**: Apply feature transformers for clustering analysis
5. **Clustering**: Apply K-means with hyperparameter optimization (Bayesian optimization trials)
6. **Evaluation**: Assess using silhouette analysis, Calinski-Harabasz, and Davies-Bouldin scores
7. **Analysis**: Generate pairplots of relevant features to visualize cluster separation

**Data Requirements:**
MLNav works with **offline processed data** that has already been computed by the Position Engine in a separate stage. Input files must follow this format:

```
# Expected format: AI_Multipath_*.txt files
# Year  Month  Day  Hour  Minute  Second   Sat_id           Freq   Code   Phase   Doppler   Snr   Elevation   Residual   Iono   [Multipath]
  2019      8   14     6      59  52.100      E27   1590939914.0   13.0    13.0      14.0   0.0        67.0       62.0   62.0           [5]
  2019      8   14     6      59  52.200      E24   1575925548.0   66.0    66.0      67.0   0.0       144.0       61.0   61.0           [3]
...
```

### RLNav Pipeline
**Reinforcement Learning approach for adaptive GNSS observation selection:**

**Objective**: Uses PPO (Proximal Policy Optimization) agent to dynamically select which GNSS observations to use in the positioning algorithm, optimizing accuracy through adaptive measurement weighting.

**Key Components:**
- **Observation Selection**: Agent receives all observations from current epoch and decides which measurements to activate/deactivate for the positioning algorithm
- **Multi-Objective Optimization**: Rewards based on improvement ratios for NEU (North-East-Up) error components
- **Curriculum Learning**: Progressive difficulty increase using scenarios ordered by positioning error difficulty
- **Parallel Training**: Multiple environments for efficient experience collection

**Technical Details:**
- **Reward Function**: Logarithmic improvement ratio centered at 0:
  ```
  r = log(e_baseline/e_agent)     if e_agent < e_baseline
  r = -log(e_agent/e_baseline)    if e_agent ≥ e_baseline
  ```
  where `e_agent` is agent's Euclidean error and `e_baseline` is baseline PE error

- **Scenario Difficulty**: Based on baseline PE error:
  - **Low difficulty**: <1m error (LOWH_LOWV labels)
  - **Medium difficulty**: 1-3m error (MIDH_MIDV labels) 
  - **High difficulty**: >3m error (HIGHH_HIGHV labels)

- **Training Strategy**:
  - **Parallel Environments**: 20 parallel environments for experience collection
  - **Active Environments**: 10 active environments per iteration
  - **RNN Support**: Optional recurrent networks for temporal dependencies
  - **Continuous Evaluation**: Cascaded environments for ongoing assessment

### Shared Components
- **NavUtils**: Centralized logging, configuration management, and utility functions
- **PEWrapper**: Position Engine integration layer using ctypes for C++ library communication
- **Data Management**: DVC-based data versioning with comprehensive pipeline management

### Position Engine Integration

> **Important**: Only **RLNav** directly interacts with the Position Engine. **MLNav** works with offline processed data that was previously computed by the PE.

The Position Engine (PE) is the core positioning module handling all GNSS positioning operations. **RLNav** interfaces with proprietary GMV positioning software through:

- **Dynamic Library Loading**: PE compiled as shared library `libcommon_lib_PE_develop.so`
- **Environment Variable**: Library path set via `LD_LIBRARY_PATH`
- **API Interface**: Complete C++ API wrapped using Python ctypes
- **Real-time Data Flow**: Supports RTCM, UBX, and SBF protocol parsing and processing

---

## 📁 Project Structure

```
NavAI/
├── 📁 config/               # Configuration files
│   ├── MLNav/
│   │   └── params.yaml      # ML pipeline parameters
│   └── RLNav/
│       └── params.yaml      # RL pipeline parameters
├── 📁 data/                 # Data directories (DVC managed)
│   ├── MLNav/               # ML pipeline data
│   │   ├── raw/             # Raw GNSS data
│   │   ├── processed/       # Processed datasets
│   │   └── normalized/      # Normalized features
│   ├── RLNav/               # RL pipeline data
│   │   ├── processed/       # Processed RL data
│   │   └── transformed/     # Environment-specific data
│   └── scenarios/           # Test scenarios (DVC tracked)
├── 📁 docker/               # Docker configurations
│   ├── MLNav.Dockerfile     # ML environment
│   └── RLNav.Dockerfile     # RL environment
├── 📁 models/               # Trained models (git ignored)
├── 📁 notebooks/            # Jupyter notebooks (git ignored)
├── 📁 reports/              # Generated reports and metrics
├── 📁 src/                  # Source code
│   ├── 🧠 mlnav/           # Machine Learning pipeline
│   │   ├── data/           # Data processing
│   │   ├── evaluate/       # Evaluation tools
│   │   ├── features/       # Feature engineering
│   │   ├── pipes/          # Pipeline components
│   │   ├── report/         # Reporting utilities
│   │   ├── train/          # Training modules
│   │   ├── types/          # Type definitions
│   │   └── utils/          # Utility functions
│   ├── 🎮 rlnav/           # Reinforcement Learning pipeline
│   │   ├── agent/          # RL agents (PPO)
│   │   ├── data/           # Data processing and datasets
│   │   ├── drivers/        # RL execution driver
│   │   ├── env/            # Environment definitions
│   │   ├── managers/       # Reference, reward and wrapper managers
│   │   ├── networks/       # Neural network architectures
│   │   ├── pipes/          # Pipeline components
│   │   ├── recoder/        # Recording utilities
│   │   ├── reports/        # Reporting and metrics
│   │   ├── types/          # Type definitions
│   │   ├── utils/          # Utility functions
│   │   ├── ppo_eval.py     # PPO evaluation script
│   │   └── ppo_train.py    # PPO training script
│   ├── 🔧 navutils/        # Shared utilities
│   │   ├── logger.py       # Logging system
│   │   ├── config.py       # Configuration management
│   │   └── singleton.py    # Design patterns
│   └── 🔌 pewrapper/       # Position Engine wrapper
├── 📁 test/                 # Unit and integration tests
├── 📄 pyproject.toml        # Project configuration
├── 📄 mlnav.dvc.yaml        # DVC - ML pipeline definition
└── 📄 README.md             # This file
```

---

## ⚙️ Installation

### Prerequisites

- **Python**: 3.10 (strictly required)
- **Poetry**: For dependency management
- **Docker**: For containerized development (optional)
- **DVC**: For data versioning
- **Git**: For version control

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/AlvaroTena/NavAI.git
cd NavAI

# Install dependencies
pip install -e .

# Setup DVC (if using data versioning)
dvc pull
```

### Development Setup

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project in development mode
poetry install

# Install with optional ML dependencies
poetry install --extras ml

# Install with optional RL dependencies  
poetry install --extras rl

# Install all optional dependencies
poetry install --extras "ml rl"

# Run commands in virtual environment
poetry run <command>
```

### Environment Setup

For detailed environment configuration (`.env` files, thread optimization variables), see the **[🔧 Configuration](#-configuration)** section below.

**Quick Summary**:
- **RLNav**: Requires `.env` file with Position Engine and Neptune variables
- **MLNav**: Optionally uses thread optimization exports for performance

---

## 🚀 Usage

### MLNav Pipeline

The MLNav pipeline uses **DVC (Data Version Control)** to manage the machine learning workflow with automated stage dependencies.

#### Complete Pipeline Execution

```bash
# Run the complete MLNav pipeline
dvc repro mlnav.dvc.yaml
```

> ⚠️ **Important**: The DVC file must be named exactly `dvc.yaml` for some DVC operations. If using `mlnav.dvc.yaml`, always specify the filename explicitly.

#### Individual Pipeline Stages

The MLNav pipeline consists of the following stages (executed automatically by DVC):

1. **Process Data**: Convert raw GNSS data to processed format
2. **Data Reports**: Generate data visualization and analysis
3. **Normalize Data**: Standardize features and create transformers
4. **Train Model**: K-means clustering with Bayesian hyperparameter optimization
5. **Evaluate Model**: Performance metrics calculation
6. **Predict Data**: Apply trained model to generate predictions
7. **Model Reports**: Generate clustering analysis and visualizations

#### Manual Stage Execution

```bash
# Process raw GNSS data
python3 src/mlnav/pipes/process_data.py -c config/MLNav/params.yaml -g DEBUG

# Generate data reports
python3 src/mlnav/pipes/data_reports.py -c config/MLNav/params.yaml -g DEBUG

# Normalize data
python3 src/mlnav/pipes/normalize_data.py -c config/MLNav/params.yaml -g DEBUG

# Train clustering model
python3 src/mlnav/pipes/train_model.py -c config/MLNav/params.yaml -g DEBUG

# Evaluate trained model
python3 src/mlnav/pipes/evaluate_model.py -c config/MLNav/params.yaml -g DEBUG

# Generate predictions
python3 src/mlnav/pipes/predict_data.py -c config/MLNav/params.yaml -g DEBUG

# Create model reports
python3 src/mlnav/pipes/model_reports.py -c config/MLNav/params.yaml -g DEBUG
```

### RLNav Pipeline

The RLNav pipeline provides training and evaluation modules for the PPO agent.

#### Training RL Agent

```bash
# Complete training command with all parameters
python3 src/rlnav/ppo_train.py -c config/RLNav/params.yaml -o output/ -g DEBUG

# Training with custom parsing rate (optional)
python3 src/rlnav/ppo_train.py -c config/RLNav/params.yaml -o output/ -g INFO --parsing_rate 10
```

#### Evaluating RL Agent

```bash
# Evaluate trained agent
python3 src/rlnav/ppo_eval.py -c config/RLNav/params.yaml -o output/ -g DEBUG

# Evaluation with specific debug level
python3 src/rlnav/ppo_eval.py -c config/RLNav/params.yaml -o output/ -g WARNING
```

#### Command Parameters

- `-c, --config_file`: Path to configuration YAML file
- `-o, --output_directory`: Directory for output files (scenarios/subscenarios results)
- `-g, --debug_level`: Logging level (TRACE, DEBUG, INFO, WARNING, ERROR)
- `--parsing_rate`: Optional parameter for data parsing rate control

#### Output Structure

RLNav will generate in the output directory:
- **Training outputs**: Each processed scenario or subscenario during training
- **Evaluation results**: Final evaluation metrics and performance analysis
- **Model checkpoints**: Saved agent policies and training states

### Configuration Files

Each pipeline uses its own independent YAML configuration file:

- **MLNav**: `config/MLNav/params.yaml` - Controls data paths, model parameters, and evaluation metrics
- **RLNav**: `config/RLNav/params.yaml` - Defines training scenarios, curriculum learning, and agent parameters

> **Note**: MLNav and RLNav are completely independent modules with different approaches and functionalities. They can be used separately depending on your research needs.

### Example Workflows

#### MLNav Workflow (Machine Learning Approach)

```bash
# 1. Ensure offline processed data is available in data/MLNav/raw/
# (PE processing is done outside MLNav scope)

# 2. Run complete MLNav pipeline
dvc repro mlnav.dvc.yaml

# 3. Review clustering results
ls reports/MLNav/
```

#### RLNav Workflow (Reinforcement Learning Approach)

```bash
# 1. Setup environment variables (create .env file as shown above)
# Required: LD_LIBRARY_PATH=/path/to/pe/library

# 2. Train RL agent (automatically loads .env file)
python3 src/rlnav/ppo_train.py -c config/RLNav/params.yaml -o results/training/ -g INFO

# 3. Evaluate trained agent (automatically loads .env file)
python3 src/rlnav/ppo_eval.py -c config/RLNav/params.yaml -o results/evaluation/ -g INFO
```

---

## 🐳 Docker Development

### Requirements

Ensure the base image `nav_img` is available with Position Engine dependencies.

### Building Images

```bash
cd docker

# Build MLNav environment
docker build -t mlnav_img -f MLNav.Dockerfile .

# Build RLNav environment  
docker build -t rlnav_img -f RLNav.Dockerfile .
```

### Running Containers

```bash
# Run MLNav container
docker run -it --rm -v $(pwd):/workspace mlnav_img bash

# Run RLNav container with GPU support
docker run -it --rm --gpus all -v $(pwd):/workspace rlnav_img bash
```

---

## 🔧 Configuration

### MLNav Configuration

The MLNav configuration file `config/MLNav/params.yaml` controls all aspects of the machine learning pipeline:

#### Data Pipeline Configuration

```yaml
reader:
  path: data/MLNav/raw/    # Raw GNSS data location
  n_files: -1                            # Number of files to process (-1 = all)

process:
  path: data/MLNav/processed/
  name: AI_Multipath_Processed.h5       # Processed dataset filename

normalize:
  path: data/MLNav/normalized/
  name: AI_Multipath_Normalized.h5      # Normalized dataset filename
  transformers_path: data/MLNav/transformers/  # Feature transformers storage
```

#### Model Training Configuration

```yaml
model:
  path: data/MLNav/models/
  name: model.joblib                     # Trained model filename
  
  trials:
    path: trials/
    name: trials.pkl                     # Bayesian optimization trials storage
  
  train:
    chunksize: 0                         # Data processing chunk size (0 = all data)
    hyperparameter_tunning: true        # Enable Bayesian optimization
    hyperparameter_tunning_iters: 32    # Number of optimization iterations
    
    # K-means hyperparameter grid for optimization
    param_grid:
      n_clusters: [2, 3, 4, 5, 6]       # Number of clusters to test
      init: [k-means++, random]          # Centroid initialization methods
      n_init: [auto, 5]                 # Number of random initializations
      max_iter: [100, 300, 500, 1000]   # Maximum iterations per run
      tol: [0.00001, 0.0001, 0.001, 0.01]  # Convergence tolerance
      algorithm: [lloyd, elkan]          # K-means algorithm variants
      random_state: [42]                # Random seed for reproducibility
```

#### Evaluation and Reporting

```yaml
reports:
  path: reports/MLNav/
  
  predictions:
    name: AI_Multipath_Predictions.h5   # Model predictions output

  data_plots:
    path: data_plots/                    # Data visualization plots directory

    constellation_plot: true            # Generate constellation visualizations
    box_plot: true                     # Generate box plots
    density_plot: true                 # Generate density plots

  model_plots:
    path: model_plots/                  # Model visualization plots directory

    cluster_pairplots: true            # Generate cluster pair plots
    pairplots2d: true                  # Generate 2D pair plots
    pairplots3d: false                 # Generate 3D pair plots

  evaluation:
    unsupervised_metrics:
      name: metrics.json                 # Evaluation metrics file
      silhouette: true                   # Calculate silhouette score
      silhouette_show: false            # Display silhouette plots
      silhouette_save: false            # Save silhouette plots to file
      silhouette_split: 5000            # Sample size for silhouette calculation
      calinski_harabasz: true           # Calculate Calinski-Harabasz index
      davies_bouldin_score: true        # Calculate Davies-Bouldin score
```

### RLNav Configuration

The RLNav configuration file `config/RLNav/params.yaml` defines the reinforcement learning training setup:

#### Scenario Management

```yaml
scenarios:
  path: data/scenarios/                  # Scenarios data directory
  n_scenarios: -1                        # Number of scenarios to process (-1 = all)
  skip_first: 10                        # Skip first N scenarios
  n_generations: 1                      # Number of training generations per scenario
  
  # Priority scenarios for training
  priority: [
    scenario_931_000_GPSGALBDS_CSIRELAND_L1L2_KINEMATIC_20231003,
  ]
  
  # All available subscenarios for curriculum learning
  # When curriculum_learning is enabled, these are automatically sorted by difficulty:
  # 1. First by vertical component: LOWV → MIDV → HIGHV
  # 2. Then by horizontal component: LOWH → MIDH → HIGHH
  # Final order: LOWH_LOWV → LOWH_MIDV → LOWH_HIGHV → MIDH_LOWV → MIDH_MIDV → MIDH_HIGHV → HIGHH_LOWV → HIGHH_MIDV → HIGHH_HIGHV
  subscenarios: [
    scenario_931_000_GPSGALBDS_CSIRELAND_L1L2_KINEMATIC_20231003_01_LOWH_LOWV,  # <1m error
    scenario_911_000_CS_KINEMATIC_20230227_SEMIURBAN_01_LOWH_LOWV,
    # ... more scenarios in increasing difficulty order
    scenario_913_000_CS_KINEMATIC_20230310_URBAN_13_HIGHH_HIGHV,              # >3m error
  ]
  
  subscenarios_done: 0                  # Number of subscenarios already completed
  curriculum_learning: True             # Enable automatic difficulty-based sorting
```

#### Training Configuration

```yaml
training:
  num_parallel_environments: 20         # Parallel environments for experience collection
  active_environments_per_iteration: 10 # Active environments per training iteration
  
  rnn:
    enable: True                        # Enable recurrent neural networks
    window_size: 1                     # RNN temporal window size
```

#### Evaluation Configuration

```yaml
eval:
  model_path: models/                   # Directory containing trained models
  model_name: policy           # Specific model name for evaluation
```

#### Neptune Monitoring (Optional)

```yaml
neptune:
  monitoring_times: False              # Enable Neptune time monitoring
  tensorboard: False                   # Enable TensorBoard integration
```

### Configuration Parameters Explained

#### RLNav Parameters

**Scenario Difficulty Labels** (RLNav only):
- **LOWH_LOWV**: Low horizontal error (<1m), Low vertical error (<1m) - **Easy**
- **MIDH_MIDV**: Medium horizontal error (1-3m), Medium vertical error (1-3m) - **Medium**  
- **HIGHH_HIGHV**: High horizontal error (>3m), High vertical error (>3m) - **Hard**

**Training Parameters** (RLNav only):
- **num_parallel_environments**: Number of parallel environments for experience collection
- **active_environments_per_iteration**: Active environments per training iteration
- **skip_first**: Skip first N scenarios
- **curriculum_learning**: Enable automatic difficulty-based sorting of subscenarios
- **subscenarios_done**: Number of subscenarios already completed in curriculum learning
- **n_generations**: Number of training generations per scenario
- **rnn.enable**: Enable recurrent neural networks for temporal dependencies
- **rnn.window_size**: RNN temporal window size for sequence processing
- **parsing_rate**: Optional parameter to subsample 
scenario data at specific intervals

#### MLNav Parameters

**Data Processing** (MLNav only):
- **chunksize**: Controls memory usage during processing (0 = process all data at once)
- **n_files**: Number of offline data files to process (-1 = all available files)

**Model Parameters** (MLNav only):
- **n_clusters**: Determines how many multipath patterns to identify in clustering
- **hyperparameter_tunning**: Enable Bayesian optimization for K-means parameters
- **hyperparameter_tunning_iters**: Number of optimization iterations to perform
- **param_grid**: Search space for K-means hyperparameters (init, max_iter, tol, etc.)

**Evaluation Metrics** (MLNav only):
- **silhouette**: Calculate silhouette score for cluster quality assessment
- **silhouette_show/save**: Display or save silhouette analysis plots
- **calinski_harabasz**: Calculate Calinski-Harabasz index for cluster separation
- **davies_bouldin_score**: Calculate Davies-Bouldin score for cluster compactness

### Environment Variables

#### RLNav Environment Variables

**RLNav** uses a `.env` file to manage environment variables automatically. The code loads these variables using `python-dotenv` when running RLNav pipelines.

Create a `.env` file in the project root with the following variables:

```bash
# .env.example - Copy this to .env and modify values as needed

# Position Engine Integration (Required for RLNav)
LD_LIBRARY_PATH=/path/to/pe/library

# Neptune Experiment Tracking (Required)
NEPTUNE_API_TOKEN=your_neptune_api_token_here
NEPTUNE_CUSTOM_RUN_ID=your_custom_run_id
NEPTUNE_MODE=debug
```

**RLNav Variable Descriptions:**

- **`LD_LIBRARY_PATH`** *(Required)*: Path to directory containing `libcommon_lib_PE_develop.so`
- **`NEPTUNE_API_TOKEN`** *(Required)*: API token for Neptune experiment tracking and monitoring
- **`NEPTUNE_CUSTOM_RUN_ID`** *(Optional)*: Custom run identifier for Neptune experiments
- **`NEPTUNE_MODE`** *(Optional)*: Set to `"debug"` to prevent uploading artifacts to Neptune.ai. In debug mode, Neptune automatically disables artifact uploads and the code includes additional checks to skip metric calculations for enhanced security.

> **Note**: The `.env` file is automatically loaded by RLNav scripts (`ppo_train.py`, `ppo_eval.py`) using `load_dotenv(override=True)`.

#### MLNav Performance Optimization

**MLNav** uses thread optimization variables for ML performance. These can be configured in two ways:

**Option 1: DVC Pipeline Configuration (Automatic)**
```yaml
# In mlnav.dvc.yaml - MLNav pipeline configuration
env_variables:
  cmd: export OPENBLAS_NUM_THREADS=64 && export NUM_THREADS=64 && export OMP_NUM_THREADS=64
  always_changed: true
```

**Option 2: Manual Export (Recommended)**
```bash
# Export variables manually before running DVC pipeline
export OPENBLAS_NUM_THREADS=64
export NUM_THREADS=64  
export OMP_NUM_THREADS=64

# Then run MLNav pipeline
dvc repro mlnav.dvc.yaml
```

**MLNav Thread Variables:**
- **`OPENBLAS_NUM_THREADS=64`**: Optimizes BLAS operations for clustering algorithms
- **`NUM_THREADS=64`**: General thread count for parallel ML processing
- **`OMP_NUM_THREADS=64`**: OpenMP threads for mathematical computations

> **Important**: Manual export is recommended because the DVC DAG may execute pipeline stages in different orders. This ensures thread optimization is available regardless of execution sequence.

---

## 🔌 Position Engine Integration

### Overview

The Position Engine (PE) wrapper provides seamless integration between NavAI and the proprietary GMV positioning software. The wrapper handles all communication through a C++ shared library interface using Python ctypes.

### Library Setup

#### Prerequisites

1. **Shared Library**: Position Engine compiled dynamically
2. **Environment Variable**: Library must be accessible via `LD_LIBRARY_PATH`
3. **API Compatibility**: Wrapper and PE API versions must match

#### Library Verification

```bash
# Verify library is accessible (after configuring .env file)
ldconfig -p | grep libcommon_lib_PE_develop
```

> **Note**: Environment configuration is detailed in the **[Environment Variables](#environment-variables)** section above.

### PE API Interface

The wrapper exposes the complete Position Engine C++ API through Python:

#### Core Functions

```python
from pewrapper.api.pe_api import Position_Engine_API

# Initialize PE API
pe_api = Position_Engine_API(lib_path="/path/to/pe/library")

# Core positioning functions
pe_api.Reboot(config_info, flag)                  # Initialize PE with configuration
pe_api.LoadGnssMessage(msg, msg_length)           # Load GNSS data (RTCM/UBX/SBF)
pe_api.PreCompute(gm_time, pvt_output, features)  # Extract features for ML/RL
pe_api.Compute(state, pvt_output, output)         # Compute position solution
pe_api.LoadPredictionsAI(predictions)             # Load AI predictions/decisions
```

#### Message Format Support

The wrapper automatically handles multiple GNSS data formats:

- **RTCM**: Radio Technical Commission for Maritime Services
- **UBX**: u-blox proprietary format
- **SBF**: Septentrio Binary Format

#### RLNav Data Flow Integration

```python
# RLNav integration based on evaluation pipeline
def rlnav_evaluation_flow():
    # 1. Load environment and configuration
    load_dotenv(override=True)
    config = load_config("config/RLNav/params.yaml")
    
    # 2. Initialize scenario management
    scenarios = load_scenarios_list(config.scenarios.path, 
                                   config.scenarios.skip_first,
                                   config.scenarios.n_scenarios,
                                   config.scenarios.priority)
    
    wrapper_mgr = WrapperManager(scenarios, config.scenarios.path, 
                                output_path, reward_mgr, npt_run)
    
    # 3. Load trained RL policy
    policy = tf.saved_model.load(os.path.join(config.eval.model_path, 
                                             config.eval.model_name))
    
    # 4. Process each scenario
    while wrapper_mgr.next_scenario(parsing_rate=parsing_rate):
        # Create RL environment with Position Engine
        pe_env = PE_Env(
            configMgr=wrapper_mgr.configMgr,
            wrapper_data=wrapper_mgr.wrapper_data,
            rewardMgr=wrapper_mgr.rewardMgr,
            min_values=min_values,
            max_values=max_values,
            transformers_path=config.transformed_data.path,
            eval_mode=True
        )
        
        # Initialize environment
        time_step = pe_env.reset()
        policy_state = policy.get_initial_state(batch_size=1)
        
        # Main evaluation loop
        while not pe_env.is_done():
            # Get RL agent action (observation selection)
            policy_step = policy.action(time_step, policy_state)
            
            # Execute action in PE environment
            time_step = pe_env.step(policy_step.action)
            policy_state = policy_step.state
        
        pe_env.close()
```

**Core PE Environment Processing Logic:**

The PE_Env.step() method implements the core interaction cycle between the RL agent and Position Engine:

1. **Epoch Processing**: The wrapper processes the next GNSS data epoch and determines if agent action is needed
2. **Feature Extraction**: When action is required, GNSS features (satellite observations, elevations, residuals, etc.) are extracted from the Position Engine
3. **Data Transformation**: Raw features are processed through trained transformers to create normalized observations for the RL agent
4. **Action Application**: The agent's binary decisions (select/discard observations) are loaded into the Position Engine
5. **Position Computation**: The Position Engine computes the final position using only the agent-selected observations
6. **Reward Calculation**: The positioning performance is compared against the baseline to generate training rewards

This cycle continues until all scenario data is processed, with the agent learning to select optimal GNSS observations for improved positioning accuracy.

#### MLNav Offline Data Processing

```python
# MLNav offline processing with pre-computed data
from mlnav.data.reader import Reader

def mlnav_process_offline_data(data_directory):
    # 1. Read pre-processed PE data files (AI_Multipath_*.txt)
    reader = Reader()
    dataframes = reader.perform_reading(data_directory, n_files=-1)
    
    # 2. Concatenate all scenarios
    combined_data = pd.concat(dataframes, ignore_index=True)
    
    # 3. Apply clustering analysis
    cluster_predictions = mlnav_model.predict(combined_data)
    
    return cluster_predictions
```

### Data Management

#### RLNav Wrapper Data Manager

The `WrapperDataManager` handles real-time scenario data parsing and processing for **RLNav only**:

```python
from pewrapper.managers.wrapper_data_mgr import WrapperDataManager

# Initialize data manager
data_mgr = WrapperDataManager(
    initial_epoch=start_time,
    final_epoch=end_time,
    configMgr=config_manager
)

# Parse scenario file
success, info = data_mgr.parse_wrapper_file("scenario.txt", parsing_rate=1)

# Iterate through epochs
# Option 1: Using iterator with next()
wrapper_iterator = data_mgr.get_iterator()
try:
    while True:
        epoch, epoch_data = next(wrapper_iterator)
        # Process each epoch with PE API
        # Note: process_gnss_epoch is not a real function, it's a placeholder
        # for the actual data loading and processing logic found in wrapper.py
        process_gnss_epoch(data_mgr, pe_api, epoch_data)
except StopIteration:
    pass

# Option 2: Using __iter__ with for loop
for epoch, epoch_data in data_mgr:
    # Process each epoch with PE API
    # Note: process_gnss_epoch is not a real function, it's a placeholder
    # for the actual data loading and processing logic found in wrapper.py
    process_gnss_epoch(data_mgr, pe_api, epoch_data)
```

#### RLNav Expected Data Format

RLNav input scenario files should follow this structure for real-time processing:

```
# Date      Time              Type      HexData/Measurements
2023/10/03  12:00:01.000000   GNSS      b5620215d0065...  # Raw GNSS messages in hex
2023/10/03  12:00:01.000000   COMPUTE   Process_compute 0.000000 # PE computation trigger
2023/10/03  12:00:01.001000   IMU       timestamp,acceleration_x,acceleration_y,acceleration_z,angular_velocity_x,angular_velocity_y,angular_velocity_z
```

#### MLNav Expected Data Format

MLNav works with **offline processed PE data** in text files (`AI_Multipath_*.txt`):

```
# Pre-processed by Position Engine outside MLNav scope
# Year  Month  Day  Hour  Minute  Second   Sat_id           Freq   Code   Phase   Doppler   Snr   Elevation   Residual   Iono   [Multipath]
  2019      8   14     6      59  52.100      E27   1590939914.0   13.0    13.0      14.0   0.0        67.0       62.0   62.0           [5]
  2019      8   14     6      59  52.200      E24   1575925548.0   66.0    66.0      67.0   0.0       144.0       61.0   61.0           [3]
...
```

### Error Handling and Logging

The wrapper provides comprehensive error handling and logging:

```python
from navutils.logger import Logger

# Logging integration
Logger.log_message(
    Logger.Category.DEBUG,
    Logger.Module.PE,
    "Position Engine initialized successfully"
)

# API version verification
if pe_api_version != wrapper_version:
    Logger.log_message(
        Logger.Category.ERROR,
        Logger.Module.PE,
        f"Version mismatch: PE={pe_api_version}, Wrapper={wrapper_version}"
    )
```

---

## 📊 Data Pipeline (DVC)

### Pipeline Overview

The MLNav pipeline uses **DVC (Data Version Control)** to manage the complete machine learning workflow with automatic dependency tracking and reproducible execution.

### Pipeline Stages

The `mlnav.dvc.yaml` file defines 7 interconnected stages:

#### Stage Details

1. **Environment Variables**
   ```yaml
   env_variables:
     cmd: export OPENBLAS_NUM_THREADS=64 && export NUM_THREADS=64 && export OMP_NUM_THREADS=64
     always_changed: true
   ```
   > **Note**: This is **MLNav-specific** DVC pipeline configuration for ML performance optimization. Manual export of these variables before running DVC is recommended to ensure availability regardless of DAG execution order.

2. **Process Data**
   - **Input**: Raw GNSS scenarios (`${reader.path}`)
   - **Output**: Processed HDF5 dataset (`AI_Multipath_Processed.h5`)
   - **Function**: Convert raw GNSS data to structured format

3. **Data Reports**
   - **Input**: Processed dataset
   - **Output**: Data visualization plots
   - **Function**: Generate constellation, box, and density plots

4. **Normalize Data**
   - **Input**: Processed dataset
   - **Output**: Normalized dataset + feature transformers
   - **Function**: Standardize features for clustering

5. **Train Model**
   - **Input**: Normalized dataset
   - **Output**: Trained K-means model + optimization trials
   - **Function**: Hyperparameter tuning and model training

6. **Evaluate Model**
   - **Input**: Trained model
   - **Output**: Evaluation metrics (JSON)
   - **Function**: Calculate clustering performance metrics

7. **Predict Data**
   - **Input**: Processed data + trained model + transformers
   - **Output**: Model predictions (HDF5)
   - **Function**: Apply clustering to generate predictions

8. **Model Reports**
   - **Input**: Predictions + processed data
   - **Output**: Clustering analysis plots
   - **Function**: Generate pairplots and cluster visualizations

### Pipeline Execution

```bash
# Complete pipeline
dvc repro mlnav.dvc.yaml

# Specific stage
dvc repro mlnav.dvc.yaml:train_model

# Force re-run (ignore cache)
dvc repro --force mlnav.dvc.yaml

# Dry-run (show what would be executed)
dvc repro --dry mlnav.dvc.yaml
```

### Data Versioning

DVC manages data versioning and caching:

- **Cache**: Large datasets cached locally and remotely
- **Push**: Upload processed data to remote storage
- **Pull**: Download required data for reproduction
- **Tracking**: Git tracks pipeline definitions, DVC tracks data

---

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src

# Run specific test module
poetry run pytest test/mlnav/

# Run with verbose output
poetry run pytest -v
```

---

## 📈 Monitoring and Logging

NavAI includes comprehensive monitoring and logging:

- **Neptune Integration**: Experiment tracking and visualization with metrics logged to app.neptune.ai
- **TensorBoard Support**: For RL training monitoring (integrated with Neptune)
- **Structured Logging**: Centralized logging system via `navutils.logger`
- **Metrics Reporting**: Automated performance metrics generation

```bash
# View logs
tail -f pe_log_files/wrapper_session.log
```

---

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow code style**: Use black formatter
4. **Add tests**: Ensure new features have corresponding tests
5. **Update documentation**: Keep README and docstrings current
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Guidelines

- Follow [Conventional Commits](https://conventionalcommits.org/) for commit messages
- Use [Conventional Branch](https://conventional-branch.github.io/) naming (e.g., `feature/new-algorithm`, `bugfix/training-issue`)
- Write comprehensive docstrings for all functions and classes
- Add type hints where applicable
- Ensure all tests pass before submitting PRs

---

## 📚 Publications

The research and methodologies implemented in NavAI have been published in peer-reviewed conference proceedings:

### MLNav (Machine Learning Pipeline)

**Tena, Á., Chamorro, A., & Calle, J. D. (2025). Enhancing GNSS PPP Algorithms with AI: Towards Mitigating Multipath Effects. Engineering Proceedings, 88(1), 56. https://doi.org/10.3390/engproc2025088056**

This paper presents the lightweight machine learning approach using K-means clustering for multipath mitigation in GNSS measurements, trained on ~50 hours of diverse environmental data.

### RLNav (Reinforcement Learning Pipeline)

**"Reinforcement Learning-Driven GNSS Observation Selection For Enhanced PPP Accuracy"** - Presented at ENC25, paper pending publication.

This work details the PPO-based agent for adaptive GNSS observation selection with multi-objective optimization and curriculum learning strategies.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>Made with ❤️ for GNSS Navigation Optimization</strong></p>
</div>

