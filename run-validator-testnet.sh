NAME=${VAL_PM2_NAME:-"testnet_chunking_validator"}

VAL_PM2_NAME=$NAME bash run-validator.sh --subtensor.chain_endpoint test --subtensor.network test --wandb.project_name chunking-testnet --netuid 166 "$@" 
