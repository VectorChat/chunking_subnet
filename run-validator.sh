#!/bin/bash
source .env

NAME=chunking_validator_autoupdate

if pm2 describe $NAME > /dev/null 2>&1; then
    echo "Process '$NAME' is running. Stopping it..."
    pm2 stop $NAME
    pm2 delete $NAME
fi

check_package_installed() {
    local package_name="$1"
    os_name=$(uname -s)
    
    if [[ "$os_name" == "Linux" ]]; then
        # Use dpkg-query to check if the package is installed
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
        echo "Unknown operating system"
        return 0
    fi
}

check_package_installed "jq"
if [ $? -eq 0 ]; then
    echo "jq is not installed. Please install jq before running this script."
    exit 1
fi

pm2 start validator-autoupdate.sh --name $NAME --max-restarts 5 -- --netuid $NETUID --wallet.name $COLDKEY --wallet.hotkey $HOTKEY --log_level debug --openaikey $OPENAI_API_KEY "$@"