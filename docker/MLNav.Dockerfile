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
      wget\
      cmake \
      g++ \
   && rm -rf /var/lib/apt/lists/*
   
RUN wget https://github.com/libgit2/libgit2/archive/refs/tags/v1.5.0.tar.gz -O libgit2-1.5.0.tar.gz \
    && tar xzf libgit2-1.5.0.tar.gz\
    && cd libgit2-1.5.0 \
    && cmake . \
    && make \
    && make install

RUN wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_10_10.tar.gz \
    && tar xzf hdf5-1_10_10.tar.gz \
    && cd hdf5-hdf5-1_10_10 \
    && ./configure --prefix=/usr/local/hdf5 --enable-cxx \
    && make \
    && make check \
    && make install \
    && make check-install

RUN apt update \
    && apt install -y --no-install-recommends \
    python3\
    python3-pip\
    python3-dev\
    pipx\
    gdb \
    ssh \
    htop

RUN pipx ensurepath
RUN pipx install poetry

WORKDIR /ml_nav
COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-interaction --no-ansi
RUN poetry install --extras ml --no-interaction --no-ansi

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