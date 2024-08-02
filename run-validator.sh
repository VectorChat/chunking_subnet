#!/bin/bash
source .env
pm2 start validator-autoupdate.sh --name chunking_validators_autoupdate --cron-restart="$CRON_SCHEDULE" -- --netuid $NETUID --wallet.name $COLDKEY --wallet.hotkey $HOTKEY --log_level debug --openaikey $OPENAI_API_KEY