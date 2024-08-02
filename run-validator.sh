#!/bin/bash
source .env

bash validator-autoupdate.sh --netuid $NETUID --wallet.name $COLDKEY --wallet.hotkey $HOTKEY --log_level debug --openaikey $OPENAI_API_KEY