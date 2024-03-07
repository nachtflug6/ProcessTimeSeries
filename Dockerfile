FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive


# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3-pip \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

 # Create a working directory.
RUN mkdir /app
WORKDIR /app

# Install any python packages you need
COPY requirements_dev.txt requirements_dev.txt

RUN python3 -m pip install -r requirements_dev.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set the entrypoint
ENTRYPOINT [ "python3" ]