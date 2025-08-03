#!/bin/bash

# Set default port if not specified
PORT=${PORT:-8000}
IMAGE_NAME="housing-api"

echo "Starting deployment of $IMAGE_NAME on port $PORT"

# Stop any container already using the port
EXISTING_CONTAINER=$(docker ps -q --filter "ancestor=$IMAGE_NAME")
if [ -n "$EXISTING_CONTAINER" ]; then
    echo "Stopping existing container"
    docker stop $EXISTING_CONTAINER
fi

# Run container
docker run -d -p $PORT:8000 --name housing_api_container $IMAGE_NAME

# Check if container is running
sleep 2
if docker ps | grep housing_api_container > /dev/null; then
    echo "$IMAGE_NAME deployed successfully at http://localhost:$PORT"
else
    echo "Deployment failed"
fi
