pm2 delete chunking_validator_autoupdate
pm2 delete chunking_validators_main_process

source .env

# Define color codes
CYAN='\033[36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}Would you like to delete the state file? ${YELLOW}(y/n)${NC}"

read delete_state

if [ "$delete_state" == "y" ]; then
    STATE_DELETE_PATH=~/.bittensor/miners/$COLDKEY/$HOTKEY/netuid$NETUID/validator/state.npz
    rm -f $STATE_DELETE_PATH
    echo "Deleted state file at $STATE_DELETE_PATH"
fi

echo -e "\n${CYAN}Would you like to restart the validator? ${YELLOW}(y/n)${NC}"

read start_validator

if [ "$start_validator" == "y" ]; then
    bash run-validator.sh     
fi