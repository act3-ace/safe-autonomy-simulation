ARG PIP_INDEX_URL
ARG POETRY_VERSION=1.5.0

FROM docker.io/python:3.10.5 as develop

#########################################################################################
# develop stage contains base requirements. Used as base for all other stages.
#
#   (1) APT Install deps for the base development only - i.e. items for running code
#   (2) Install the repository requirements
#   (3) logs file created
#
#########################################################################################

# Re-declare ARGs
ARG PIP_INDEX_URL
ENV DEBIAN_FRONTEND=noninteractive

# Run basic app installs here
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    git \
    host \
    vim \
    net-tools \
    openssh-server \
    iproute2 \
    inetutils-ping \
    # python3-dev \
    # python-is-python3 \
    # python3-pip \
    # python3-venv \
    # libcudnn8=8.1.1.33-1+cuda11.2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git=1:2.30.2-1* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Poetry
# TODO create method for installing in 620
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local POETRY_VERSION=${POETRY_VERSION} /usr/local/bin/python - 

# Test that poetry is installed correctly
RUN poetry --version

WORKDIR /opt/project
COPY . ./
RUN poetry config virtualenvs.create false && poetry install --without docs,lint,test

#########################################################################################
# Build stage packages from the source code
#########################################################################################
FROM develop as build
ENV ROOT=/opt/libsafe-autonomy-simulation
ARG PIP_INDEX_URL
WORKDIR /opt/project

RUN poetry build && mv dist/ ${ROOT}

#########################################################################################
# the package stage contains everything required to install the project from another container build
#########################################################################################
FROM scratch as package
ENV ROOT=/opt/libsafe-autonomy-simulation
COPY --from=build ${ROOT} ${ROOT}

#########################################################################################
# the CI/CD pipeline uses the last stage by default so set your stage for CI/CD here with FROM your_ci_cd_stage as cicd
# this image should be able to run and test your source code
# python CI/CD jobs assume a python executable will be in the PATH to run all testing, documentation, etc.
#########################################################################################
FROM build as cicd

chmod +x release.sh
RUN poetry install --only test,docs,lint