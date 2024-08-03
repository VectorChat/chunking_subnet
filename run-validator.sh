#!/bin/bash
source .env

NAME=chunking_validator_autoupdate

if pm2 describe $NAME > /dev/null 2>&1; then
    echo "Process '$NAME' is running. Stopping it..."
    pm2 stop $NAME
    pm2 delete $NAME
fi

pm2 start validator-autoupdate.sh --name $NAME -- --netuid $NETUID --wallet.name $COLDKEY --wallet.hotkey $HOTKEY --log_level debug --openaikey $OPENAI_API_KEY