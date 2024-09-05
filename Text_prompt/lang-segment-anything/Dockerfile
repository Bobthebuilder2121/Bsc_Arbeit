FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
# prevent apt from hanging
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
     SETUPTOOLS_USE_DISTUTILS=stdlib

RUN conda update conda -y

# Install libraries in the brand new image. 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         git \
         python3-opencv \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# create workspace
ENV HOME /workspace
WORKDIR $HOME

RUN conda install -c "nvidia/label/cuda-12.1.1" cuda -y
ENV CUDA_HOME=$CONDA_PREFIX
ENV PATH=/usr/local/cuda/bin:$PATH

# dependency: lang-segment-anything
RUN apt update

# installing system dependencies:
RUN apt install -y git
RUN apt install libgl1-mesa-glx -y
RUN apt install libglib2.0-0 -y

# copy source code:
COPY . $HOME/lang-segment-anything

# installing python dependencies:
WORKDIR $HOME/lang-segment-anything
RUN pip install -e .

# running the basic test,
# then it will held the weights inside the image,
# so no "cold start"
RUN python running_test.py

# running the app:
CMD ["lightning", "run", "app", "app.py"]