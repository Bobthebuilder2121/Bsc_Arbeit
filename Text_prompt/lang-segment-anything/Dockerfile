# Base image with a minimal Conda setup
FROM continuumio/miniconda3:latest

# Prevent apt from hanging
ARG DEBIAN_FRONTEND=noninteractive

# Update system and install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    python3-opencv \
    ca-certificates \
    software-properties-common \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc-11 \
    g++-11 \
    && rm -rf /var/lib/apt/lists/*

# Set the default GCC and G++ version
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11

# Set Python version explicitly to 3.10 to ensure compatibility with PyTorch 2.1.0
RUN conda install python=3.10 -y

# Install CUDA directly into the base environment
RUN conda install -c "nvidia/label/cuda-12.1.1" cuda -y

# Install PyTorch, torchvision, and PyTorch3D with CUDA support in the base environment
RUN conda install -c pytorch -c nvidia \
    pytorch=2.1.0 \
    torchvision \
    torchaudio \
    pytorch-cuda=12.1 \
    -y && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y && \
    conda install pytorch3d=0.7.5 -c pytorch3d

# Set CUDA_HOME environment variable to the base environment location
ENV CUDA_HOME /opt/conda/

# Add CUDA to PATH and LD_LIBRARY_PATH
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set additional environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV SETUPTOOLS_USE_DISTUTILS=stdlib

# Install additional dependencies using pip for packages not available in Conda
RUN conda run -n base python -m pip install hydra-core \
    omegaconf \
    opencv-python \
    einops \
    visdom \
    tqdm \
    scipy \
    plotly \
    scikit-learn \
    imageio[ffmpeg] \
    gradio \
    trimesh \
    huggingface_hub \
    numpy==1.26.3 \
    pycolmap==3.10.0 \
    pyceres \
    poselib==2.0.2

# Install LightGlue
RUN git clone https://github.com/jytime/LightGlue.git dependency/LightGlue && \
    cd dependency/LightGlue && \
    conda run -n base python -m pip install -e . && \
    cd ../../

# Create workspace
ENV HOME /workspace
WORKDIR $HOME

# Copy source code
COPY . $HOME/lang-segment-anything

# Install project dependencies in the base environment
WORKDIR $HOME/lang-segment-anything
RUN conda run -n base python -m pip install -e .

# Run the basic test
RUN conda run -n base python running_test.py

# Run the app
CMD ["lightning", "run", "app", "app.py"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.