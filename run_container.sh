#!/bin/bash

#if ! command -v nvidia-smi &> /dev/null
# then
#	echo "NVIDIA drivers are not installed. Exiting..."
#	exit 1
# fi

if ! dpkg -l | grep -q nvidia-container-toolkit; then
	echo "NVIDIA Container Toolkit was not found. Installing..."

	sudo apt-get update
	sudo apt-get install -y nvidia-docker2
	sudo apt -y install docker.io nvidia-container-toolkit && \
		sudo systemctl daemon-reload && \
		sudo systemctl restart docker

	echo "NVIDIA Container Toolkit and Docker installed and configured."
else
	echo "NVIDIA Container Toolkit is already installed."

fi

# Pull image
echo "Pulling image from Docker Hub..."
docker pull lucaslibshutz/scratch:latest

echo "Starting Docker container with Jupyter notebook..."

docker run --gpus all --rm -p 8888:8888 -v $(pwd)/:/workspace lucaslibshutz/scratch:latest \
	bash -c "pip install jupyter && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"

echo "Jupyter Notebook is running. You can connect to it from VS Code."
