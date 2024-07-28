#!/bin/bash

# Check Python version
if ! command -v python3 &>/dev/null || ! python3 -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ required'"; then
    echo "Error: Python 3.8 or higher is required."
    exit 1
fi

# Clone the repository
git clone https://github.com/VectorChat/chunking_subnet
cd chunking_subnet

# Install the package
pip3 install -e .

# Install punkt model
python3 -c "import nltk; nltk.download('punkt')"

echo "Setup completed successfully!"