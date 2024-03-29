# safe-autonomy-simulation Developer Guide

## Design Patterns

<!-- Describe code organization and style -->

## Building

## Building with Docker

The library provides a [Dockerfile](../Dockerfile) for building Docker containers.

### Requirements

Docker is required to build the Dockerfile using `docker compose`.

> Install Docker: <https://docs.docker.com/get-docker>

Building the Docker containers requires access to private Python packages and Git repositories. The Dockerfile is dependent on secrets containing authentication for private dependencies. This repository is configured to automate the creation of these secrets.

To utilize the automated secret creation, follow the process below:

1. Install the required tools:
   - direnv loads environment variables when entering this repository: [Install direnv](https://direnv.net/docs/installation.html)
   - crane is used to retrieve registry credentials from your system: [Install crane](https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md#installation)
   - yq is used to parse the registry credentials: [Install yq](https://github.com/mikefarah/yq/#install)
2. Hook direnv into your shell: [Setup direnv](https://direnv.net/docs/hook.html)
3. Log into the container registry "reg.git.act3-ace.com"
   - Authenticate with ACT3 Login: [ACT3 Login documentation](https://www.git.act3-ace.com/onboarding/set-up-tools/#act3-login-script)
   - Authenticate with a registry tool (docker/oras/crane/etc)
     - [`docker login`](https://docs.docker.com/reference/cli/docker/login/)
     - [`podman login`](https://docs.podman.io/en/stable/markdown/podman-login.1.html)
     - [`oras login`](https://oras.land/docs/commands/oras_login/)
     - [`crane auth login`](https://github.com/google/go-containerregistry/blob/main/cmd/crane/doc/crane_auth_login.md)
4. Navigate into the safe-autonomy-simulation directory
   - Clone the repository with `git clone`
   - Navigate into the directory with `cd`
5. Allow direnv to automatically modify your environment

   ```sh
   direnv allow
   ```

After enabling the secret creation process, whenever you enter the repository, the required environment variables will be set by direnv.

> The [`.envrc` script](../.envrc) defines the environment variables

### Build locally with Docker

Run the following command to build the library's Docker container locally:

```sh
docker compose up build
```

The [`compose.yml` file](../compose.yml) defines the "build" service and loads the requirement secrets from your environment.

## Testing

### Unit Tests

<!-- Describe how to run unit test -->

### Functional Tests

<!-- Describe how to run functional tests -->

## Releasing

The library's CI/CD pipeline is configured to create a release when a pipeline is run on the "main" branch with the variable `DO_RELEASE` set to true.

The act3-pt CLI can be used to trigger a release pipeline:

```sh
act3-pt ci run release
```
