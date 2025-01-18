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

# Set Python version explicitly to 3.9 to ensure compatibility with PyTorch 2.0.1
RUN conda install python=3.9 -y

# Install CUDA and PyTorch packages
RUN conda install -c "nvidia/label/cuda-11.8.0" cuda -y 

# Set CUDA_HOME environment variable to the base environment location
ENV CUDA_HOME /opt/conda/

# Add CUDA to PATH and LD_LIBRARY_PATH
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set additional environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV SETUPTOOLS_USE_DISTUTILS=stdlib

# Create workspace
ENV HOME /workspace
WORKDIR $HOME
