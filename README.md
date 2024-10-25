# safe-autonomy-simulation

## Intro

The `safe-autonomy-simulation` package provides a framework for building continuous time simulation environments. This package also provides an example simulation environments and simulation entities in the `safe_autonomy_simulation.sims` package.

## Installation

The `safe-autonomy-simulation` package can be installed using any python package manager.
It is recommended to install the project dependencies into an isolated virtual environment.
The following command will install `safe-autonomy-simulation` into your local environment using the `pip` package manager:

```shell
pip install safe-autonomy-simulation
```

### JAX support (experimental)

The `safe-autonomy-simulation` package supports numerical computation acceleration via the [JAX](https://jax.readthedocs.io/en/latest/index.html) library. This is an experimental feature.

The `safe_autonomy_simulation` package comes installed with JAX for CPU acceleration. JAX also provides GPU acceleration for numerical computing. If you'd like this feature the easiest way to install it is using `pip`:

```shell
pip install jax[cuda12]
```

For more information on installing JAX see the [official documentation](https://jax.readthedocs.io/en/latest/installation.html).

JAX can be enabled on a per-class basis for classes that support it. As JAX is an experimental feature it is not enabled by default.

### Installing from source

Alternatively, `safe-autonomy-simulation` can be installed from source using any of the following methods. Again, it is recommended to install this package in an isolated virtual environment. The following sections describe how to install `safe-autonomy-simulation` from source in an isolated virtual environment using `poetry`, `conda`, and `pip + virtualenv`.

#### Poetry (Recommended)

[Poetry](https://python-poetry.org/docs/) is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Poetry will automatically create an isolated virtual environment in your project location
for development.

```shell
cd safe-autonomy-simulation
poetry install
```

#### Conda

[Conda](https://conda.io/projects/conda/en/latest/index.html) is a powerful command line tool for package and environment management that runs on Windows, macOS, and Linux. Conda is often used for package and environment management in data science projects.

```shell
cd safe-autonomy-simulation
conda create -n my-env
conda activate my-env
conda install .
```

#### Pip + virtualenv

[`pip`](https://pip.pypa.io/en/stable/) is the default package installer for Python. You can use pip to install packages from the Python Package Index and other indexes. [`virtualenv`](https://virtualenv.pypa.io/en/latest/) is a tool to create isolated Python environments. Since Python 3.3, a subset of it has been integrated into the python standard library under the `venv` module. You can use `pip` together with `virtualenv` to install your project dependencies in an isolated virtual environment.

```shell
cd safe-autonomy-simulation
virtualenv venv
pip install .
```

## Local Documentation

This repository is setup to use [MkDocs](https://www.mkdocs.org/) which is a static site generator geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file.

**NOTE**: In order to properly build the documentation locally, you must first have `safe-autonomy-simulation` and its dependencies installed in your container/environment!

### Build Docs with Poetry (recommended)

Install the MkDocs modules in a container/virtual environment via Poetry:

```shell
poetry install --with docs
```

To build the documentation locally without serving it, use
the following command from within your container/virtual environment:

```shell
poetry run mkdocs build
```

To serve the documentation on a local port, use the following
command from within your container/virtual environment:

```shell
poetry run mkdocs serve 
```

## Usage

## Public Release

Approved for public release; distribution is unlimited. Case Number: AFRL-2024-3374

## Team

Jamie Cunningham,
John McCarroll,
Kyle Dunlap,
Nate Hamilton,
Charles Keating,
Kochise Bennett,
Aditesh Kumar,
Kerianne Hobbs
