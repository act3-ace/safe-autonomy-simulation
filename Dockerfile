ARG IMAGE_REPO_BASE
ARG OCI_REGISTRY=reg.git.act3-ace.com
ARG PYTHON_VERSION=3.10
ARG PIP_INDEX_URL
# renovate: datasource=pypi depName=poetry
ARG POETRY_VERSION=1.7.1
# renovate: datasource=docker depName=ace/hub/vscode-server registryUrl=reg.git.act3-ace.com
ARG ACE_VSCODE_VERSION=v0.13.6

#########################################################################################
# the develop stage contains base requirements. Used as base for all other stages.
#
#   (1) APT Install deps for the base development only - i.e. items for running code
#   (2) Install the repository requirements
#   (3) logs file created
#
#########################################################################################
FROM ${IMAGE_REPO_BASE}docker.io/python:${PYTHON_VERSION} as develop

# Re-declare ARGs
ARG POETRY_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# Run basic app installs here
# hadolint ignore=DL3008
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
#   python3-dev \
#   python-is-python3 \
#   python3-pip \
#   python3-venv \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

RUN pip install --no-cache-dir poetry==${POETRY_VERSION} \
  && poetry --version

WORKDIR /opt/project
COPY pyproject.toml poetry.lock ./

# secret mount needed for private packages
RUN --mount=type=secret,id=ACT3_SECRETS_POETRY,required=true,dst=/root/.config/pypoetry/auth.toml \
  poetry config virtualenvs.create false && \
  poetry install --without docs,lint,test,pipeline --no-root --no-cache

#########################################################################################
# the build stage packages from the source code
#########################################################################################
FROM develop as build
ENV ROOT=/opt/libtest-project

COPY . .

RUN poetry build && mv dist/ ${ROOT}

#########################################################################################
# the package stage contains everything required to install the project from another container build
#########################################################################################
FROM scratch as package
ENV ROOT=/opt/libtest-project
COPY --from=build ${ROOT} ${ROOT}

#########################################################################################
# the code-user contains everything required to run an instance of vscode in ASCE Hub
#########################################################################################
ARG ACE_VSCODE_VERSION
ARG IMAGE_REPO_BASE
ARG OCI_REGISTRY

FROM ${IMAGE_REPO_BASE}${OCI_REGISTRY}/ace/hub/vscode-server:${ACE_VSCODE_VERSION} as hub-coder-user

ARG POETRY_VERSION

WORKDIR /opt
ENV PYTHONPATH=/opt/project:${PYTHONPATH}

RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

COPY pyproject.toml poetry.lock ./

RUN --mount=type=secret,id=ACT3_SECRETS_POETRY,required=true,dst=/root/.config/pypoetry/auth.toml \
  poetry config virtualenvs.create false && \
  poetry install --with docs,lint,test,pipeline --no-root --no-cache

WORKDIR /opt

RUN --mount=type=secret,id=ACT3_SECRETS_GITLAB,required=true \
  rm -rf "/opt/safe-autonomy-simulation" \
  && git clone "https://$(cat /run/secrets/ACT3_SECRETS_GITLAB)@git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation.git"

CMD ["code-server", "--auth", "none", "--port", "8888", "--host", "0.0.0.0"]

#########################################################################################
# The "ci" stage is used to run and test your source code in CI/CD pipelines
#########################################################################################
ARG IMAGE_REPO_BASE
ARG PYTHON_VERSION

FROM ${IMAGE_REPO_BASE}docker.io/python:${PYTHON_VERSION}  as ci
ARG POETRY_VERSION

# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir poetry==${POETRY_VERSION} && \
  poetry --version

COPY pyproject.toml poetry.lock ./

RUN --mount=type=secret,id=ACT3_SECRETS_POETRY,required=true,dst=/root/.config/pypoetry/auth.toml \
  poetry config virtualenvs.create false && \
  poetry install --with lint,test,pipeline --no-root --no-cache
