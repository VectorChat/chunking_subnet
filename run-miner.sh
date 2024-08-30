#!/bin/bash
source .env

python3 -c "import bittensor as bt" > /dev/null 2>&1

if [ $? -eq 1 ]; then
    echo "Make sure your venv is activated, usually: 'source venv/bin/activate'"
    exit 1
fi

NAME=${MINER_PM2_NAME:-"chunking_miner"}

if pm2 describe $NAME > /dev/null 2>&1; then
    echo "Process '$NAME' is running. Stopping it..."
    pm2 stop $NAME
    pm2 delete $NAME
fi

pm2 start neurons/miner.py --name $NAME --cron-restart="$CRON_SCHEDULE" -- --netuid $NETUID --wallet.name $COLDKEY --wallet.hotkey $HOTKEY --log_level debug "$@"
