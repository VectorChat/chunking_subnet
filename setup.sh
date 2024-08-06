#!/bin/bash

set -e

# check Python version
if ! command -v python3 &>/dev/null || ! python3 -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ required'"; then
    echo "Error: Python 3.8 or higher is required."
    exit 1
fi

# install pm2
if ! command -v pm2 &>/dev/null; then
    echo "Installing pm2..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
    nvm install 20
    npm install -g pm2
fi

# git + dependencies
git clone https://github.com/VectorChat/chunking_subnet 
cd chunking_subnet

python3 -m venv venv
source venv/bin/activate

pip3 install -e .
python3 -c "import nltk; nltk.download('punkt')"

# .env file setup
if [ -f .env ]; then
    echo ".env file already exists. Skipping .env creation."
else
    cat << EOF > .env
NETUID=40
COLDKEY=
HOTKEY=
CRON_SCHEDULE="0 */1 * * *" # Every 1 hours
OPENAI_API_KEY=             # Required if you are a validator
WANDB_API_KEY=              # Optional if you are a validator
EOF    
fi

chmod 600 .env

echo "Setup completed successfully!"
echo "Please edit the .env file to fill in your COLDKEY and HOTKEY values, and optionally your OPENAI_API_KEY and WANDB_API_KEY if you are a looking to run a validator."