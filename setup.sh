#!/bin/bash

# Create a virtual environment
echo "Creating virtual environment..."
python3.11 -m venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install libraries from requirements.txt
echo "Installing libraries from requirements.txt..."
pip install -r requirements.txt

# Check if .env file already exists
if [ -f ".env" ]; then
    echo ".env file already exists. Skipping creation..."
else
    # Copy env backup to .env file
    if [ -f ".env.example" ]; then
        echo "Copying .env.example to .env..."
        cp .env.example .env
    else
        echo "No .env.example file found. Creating a new .env file..."
        touch .env
        echo "# Add your environment variables below" > .env
    fi
    echo "Please fill in the .env file with the necessary environment variables."
fi

echo "Setup completed successfully!"