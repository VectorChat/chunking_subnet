#!/bin/bash

set -e

check_root() {
    if [ "$(id -u)" != "0" ]; then
        echo "This script must be run as root" 1>&2
        exit 1
    fi
}

# install packages
install_packages() {
    if command -v apt-get &>/dev/null; then
        apt-get update
        apt-get install -y python3-venv python3-tk
    elif command -v yum &>/dev/null; then
        yum install -y python3-venv python3-tkinter
    elif command -v dnf &>/dev/null; then
        dnf install -y python3-venv python3-tkinter        
    else
        echo "Unable to install python3-venv. Please install it manually."
        exit 1
    fi
}

# check if running as root
check_root

# check Python version and venv availability
if ! command -v python3 &>/dev/null || ! python3 -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ required'"; then
    echo "Error: Python 3.8 or higher is required."
    exit 1
fi

install_packages

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

if [ ! -d "chunking_subnet" ]; then
    git clone https://github.com/VectorChat/chunking_subnet 
else
    echo "chunking_subnet directory already exists. Skipping git clone."
fi

cd chunking_subnet

python3 -m venv venv
source venv/bin/activate

pip3 install -e .
python3 -c "import nltk; nltk.download('punkt')"

echo "
NETUID=40
COLDKEY=
HOTKEY=
CRON_SCHEDULE=\"0 */1 * * *\" # Every 1 hours
OPENAI_API_KEY=             # Required if you are a validator
WANDB_API_KEY=              # Recommended if you are a validator
" >> .env

chmod 600 .env

GREEN='\033[0;32m'
GREY='\033[1;30m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setup completed successfully!${NC}"
echo ""
echo -e "Please edit the .env file to fill in your COLDKEY and HOTKEY values, and optionally your ${WHITE}OPENAI_API_KEY${NC} and ${WHITE}WANDB_API_KEY${NC} if you are a looking to run a validator."
echo ""
echo -e "You can now run the miner with 'bash run-miner.sh' or the validator with 'bash run-validator.sh'. Make sure to cd into the directory and activate the venv before running the scripts:"
echo ""
echo -e "${CYAN}cd chunking_subnet ${NC}"
echo -e "${CYAN}source venv/bin/activate ${NC}"