#!/bin/bash

# Build and push the Docker image
docker buildx build --platform linux/amd64 \
  -f docker/dgen/Dockerfile \
  -t us-east1-docker.pkg.dev/dgen-466702/dgen-repo-east1/dgen:latest \
  --push .

# Submit job
./submit_one.sh