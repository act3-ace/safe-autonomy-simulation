ARG PIP_INDEX_URL
ARG OCI_REGISTRY=reg.git.act3-ace.com
FROM ${OCI_REGISTRY}/act3-rl/python-poetry:v0.1.0 as develop

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

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git=1:2.30.2-1* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Poetry in image, takes a long time
# SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local/ POETRY_VERSION=1.2.0 python3 -

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

RUN poetry install --only test,docs,lint