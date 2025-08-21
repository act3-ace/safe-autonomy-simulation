#!/bin/bash -e
NEXT_RELEASE="$1"

# Update uv version if lock file exist
if [ -f uv.lock ]; then
    # Update version in pyproject.toml using sed since uv doesn't have a version command yet
    sed -i "s/^version = \".*\"/version = \"${NEXT_RELEASE}\"/" pyproject.toml
    uv export --format requirements.txt --all-groups -o requirements.dep.txt
fi

# Update Version file
echo "${NEXT_RELEASE}" >VERSION
