NAME=${VAL_PM2_NAME:-"testnet_chunking_validator"}

VAL_PM2_NAME=$NAME bash run-validator.sh --subtensor.chain_endpoint test --subtensor.network test --neuron.skip_set_weights_extrinsic --wandb.project_name chunking-testnet --netuid 166 "$@" 