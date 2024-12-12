# NavAI

## Artificial Intelligence for GNSS Optimization

NavAI focuses on Artificial Intelligence-Based Optimization of Precise Point Positioning (PPP) to achieve Adaptive Multipath Mitigation and Measurement Weighting in GNSS Positioning systems. This project leverages state-of-the-art Machine Learning and Reinforcement Learning techniques for enhanced accuracy and performance.

---

## Table of Contents

- [NavAI](#navai)
  - [Artificial Intelligence for GNSS Optimization](#artificial-intelligence-for-gnss-optimization)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Docker Development](#docker-development)
  - [Package Management](#package-management)
    - [Development Mode (Editable Installs)](#development-mode-editable-installs)
    - [Distribution Mode](#distribution-mode)

---

## Project Structure

```
NavAI/
├── config/               <- Configuration files for MLNav and RLNav
│   ├── MLNav/
│   │   └── params.yaml
│   └── RLNav/
│       └── params.yaml
├── data/                 <- Data directories managed by DVC
│   ├── MLNav/            <- Data related to Machine Learning (ML) pipeline
│   │   ├── normalized/
│   │   ├── processed/
│   │   └── raw/          <- Raw data with DVC pointers
│   ├── RLNav/            <- Data related to Reinforcement Learning (RL) pipeline
│   │   ├── processed/
│   │   └── transformed/
│   ├── scenarios/        <- Scenario definitions (DVC tracked)
│   └── scenario_update.py
├── docker/               <- Dockerfiles for environment management
├── models/               <- Model files (Git ignored)
├── notebooks/            <- Jupyter notebooks for exploratory analysis
├── reports/              <- Reports and outputs (e.g., metrics.json)
├── src/                  <- Source code for NavAI
│   ├── mlnav/            <- Code for Machine Learning pipeline
│   ├── navutils/         <- Shared utilities (e.g., logger, config)
│   ├── pewrapper/        <- Wrapper for Position Engine (PE)
│   └── rlnav/            <- Code for Reinforcement Learning pipeline
├── test/                 <- Unit and integration tests
├── pyproject.toml        <- Project configuration and dependencies
├── dvc.yaml              <- DVC pipelines for data management
├── dvc.lock              <- Locked DVC dependencies
└── README.md             <- Project documentation
```

---

## Docker Development

This project follows a modular Docker approach. Since the Position Engine (PE) code is not included in this repository, it is assumed that a base image, `nav_img`, already exists. This base image should include the foundational dependencies required for both the MLNav and RLNav pipelines. Below are the instructions for building the specific Docker images for each pipeline.

### Requirements
Ensure the base image `nav_img` is available. If it is not present, coordinate with your team to access or create it.

### Building the MLNav Docker Image
To build the Docker image for the Machine Learning (MLNav) pipeline:

```bash
cd docker
docker build -t mlnav_img -f MLNav.Dockerfile .
```

### Building the RLNav Docker Image
To build the Docker image for the Reinforcement Learning (RLNav) pipeline:

```bash
cd docker
docker build -t rlnav_img -f RLNav.Dockerfile .
```

Once the images are built, they can be used to run the respective pipelines.

---

## Package Management

A build system is used to manage Python modules. The [pyproject.toml](pyproject.toml) file specifies the build system, project configuration, and dependencies.

### Development Mode (Editable Installs)

To implement and test changes iteratively, install the project in development mode. From the root directory:

```bash
pip install -e .
```

### Distribution Mode

To build a non-editable package for distribution, use the following command from the root directory:

```bash
python3 -m build
```

