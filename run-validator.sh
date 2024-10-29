#!/bin/bash
source .env

# check if api keys are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${CYAN}OpenAI API key is not set. Please set it in the .env file as 'OPENAI_API_KEY'${NC}"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${CYAN}W&B API key is not set. Please set it in the .env file as 'WANDB_API_KEY'${NC}"
    exit 1
fi

python3 -c "import bittensor as bt" > /dev/null 2>&1

if [ $? -eq 1 ]; then
    echo "Make sure your venv is activated, usually: 'source venv/bin/activate' and that you have installed the dependencies with 'pip3 install -e .'"
    exit 1
fi

if ! python3 -c "from nltk.tokenize import sent_tokenize; sent_tokenize('Hello, world!')" > /dev/null 2>&1; then
    echo "Installing nltk stuff..."
    python3 -c "import nltk; nltk.download('punkt')" > /dev/null 2>&1
fi

NAME=${VAL_PM2_NAME:-"chunking_validator"}
AUTOUPDATE_PROC_NAME="${NAME}s_main_process"

# Define color codes
CYAN='\033[36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_package_installed() {
    local package_name="$1"
    os_name=$(uname -s)
    
    if [[ "$os_name" == "Linux" ]]; then
        if dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q "installed"; then
            return 1
        else
            return 0
        fi
    elif [[ "$os_name" == "Darwin" ]]; then
         if brew list --formula | grep -q "^$package_name$"; then
            return 1
        else
            return 0
        fi
    else
        echo "Unknown operating system for jq check"
        return 0
    fi
}

check_package_installed "jq"
if [ $? -eq 0 ]; then
    echo "jq is not installed. Please install jq before running this script."
    exit 1
fi

stop_and_delete_process() {
    local process_name="$1"
    if pm2 describe "$process_name" > /dev/null 2>&1; then
        echo -e "${CYAN}Stopping and deleting process '$process_name'...${NC}"        
        pm2 delete "$process_name"
    fi
}

# Stop and delete any existing chunking validator processes
stop_and_delete_process "$NAME"
stop_and_delete_process "${NAME}_autoupdate"
stop_and_delete_process "$AUTOUPDATE_PROC_NAME"

echo -e "\n${CYAN}Do you want to use auto-updating? ${YELLOW}(y/n)${NC}"
read use_autoupdate

SCRIPT_ARGS="--netuid $NETUID --wallet.name $COLDKEY --wallet.hotkey $HOTKEY --log_level debug $@"
export OPENAI_API_KEY=$OPENAI_API_KEY
export WANDB_API_KEY=$WANDB_API_KEY

if [ "$use_autoupdate" == "y" ]; then
    AUTO_NAME="${NAME}_autoupdate"
    echo -e "${CYAN}Starting auto-updating validator...${NC}"
    AUTOUPDATE_PROC_NAME=$AUTOUPDATE_PROC_NAME pm2 start validator-autoupdate.sh --name $AUTO_NAME --max-restarts 5 -- $SCRIPT_ARGS
else
    echo -e "${CYAN}Starting non-auto-updating validator...${NC}"
    full_command="pm2 start neurons/validator.py --name $NAME --max-restarts 5 -- $SCRIPT_ARGS"
    echo "Running command: $full_command"
    eval $full_command
fi

echo -e "\n${CYAN}Process $NAME started successfully.${NC}"
