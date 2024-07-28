#!/bin/bash

set -e

# check Python version
if ! command -v python3 &>/dev/null || ! python3 -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ required'"; then
    echo "Error: Python 3.8 or higher is required."
    exit 1
fi

git clone https://github.com/VectorChat/chunking_subnet
cd chunking_subnet

python3 -m venv venv
source venv/bin/activate

pip3 install -e .

python3 -c "import nltk; nltk.download('punkt')"

echo "Setup completed successfully!"