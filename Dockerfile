FROM docker.io/python:3.10.5 as develop

ARG PIP_INDEX_URL

#########################################################################################
# develop stage contains base requirements. Used as base for all other stages.
#
#   (1) APT Install deps for the base development only - i.e. items for running code
#   (2) Install the repository requirements
#   (3) logs file created
#
#########################################################################################

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /code
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

#########################################################################################
# Build stage packages from the source code
#########################################################################################
FROM develop as build
ENV ROOT=/opt/libsafe-autonomy-simulation
ARG PIP_INDEX_URL
WORKDIR /opt/project
COPY . .
RUN python3 -m build && mv dist/ ${ROOT}

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
