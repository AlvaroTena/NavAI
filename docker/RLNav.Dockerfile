FROM ubuntu:22.04 as boost-builder

RUN apt-get update \
    && apt-get install -y \
    libboost-dev \
    libboost-regex1.74.0 \
    libboost-system1.74.0 \
    libboost-filesystem1.74.0 \
    libboost-thread1.74.0 \
    libboost-program-options1.74.0 \
    libboost-chrono1.74.0 \
    && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y \
    wget \
    cmake \
    g++ \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=boost-builder /usr/lib/x86_64-linux-gnu/libboost* /usr/lib/x86_64-linux-gnu/

RUN wget https://github.com/libgit2/libgit2/archive/refs/tags/v1.5.0.tar.gz -O libgit2-1.5.0.tar.gz \
    && tar xzf libgit2-1.5.0.tar.gz \
    && cd libgit2-1.5.0 \
    && cmake -DBUILD_TESTS=OFF . \
    && make \
    && make install

RUN apt update \
    && apt install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gdb \
    ssh \
    htop \
    nvtop

RUN pip install tensorflow==2.15.0
RUN pip install "apache-airflow[celery]==2.9.0" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.0/constraints-3.10.txt"

RUN pip3 install -U pip
RUN pip3 install pygit2 \
    dvc \
    dvc-ssh \
    dvc-s3 \
    build \
    sortedcontainers \
    nbformat \
    toml \
    pandas \
    python-box \
    scikit-learn \
    pytest \
    tables \
    pyarrow \
    tf-agents \
    neptune \
    neptune-tensorboard \
    folium \
    pyproj \
    plotly \
    geopy \
    genv

ENV PIP_ROOT_USER_ACTION=ignore

ARG USERNAME
ARG USER_UID
ARG USER_GID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
# Set the working directory for the new user
WORKDIR /home/$USERNAME
# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

ENV OPENBLAS_NUM_THREADS=64
ENV NUM_THREADS=64
ENV OMP_NUM_THREADS=64
ENV NUMEXPR_MAX_THREADS=64