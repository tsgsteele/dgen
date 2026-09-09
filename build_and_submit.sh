#!/bin/bash

# Build and push the Docker image
docker buildx build --platform linux/amd64 --no-cache \
  -f docker/dgen/Dockerfile \
  -t us-east1-docker.pkg.dev/dgen-466702/dgen-repo-east1/dgen:latest \
  --push .

# Submit national run (all 48 states)
./submit_all.sh