#!/bin/bash
source .env
pm2 start neurons/miner.py --name chunking_miner --cron-restart="$CRON_SCHEDULE" -- --netuid $NETUID --wallet.name $COLDKEY --wallet.hotkey $HOTKEY --log_level debug