#!/bin/sh

set -e
user=ipfs

CHECK_INTERVAL=${CHECK_INTERVAL:-1}
RESTART_FILE=${RESTART_FILE:-/data/ipfs-cluster/restart}
CLUSTER_NAME=${CLUSTER_NAME:-"ipfs-cluster"}
LISTENER_URL=${LISTENER_URL:-http://listener:3000}

echo "Using listener URL: $LISTENER_URL"
echo "Using cluster name: $CLUSTER_NAME"
echo "Using leader IPFS cluster multiaddr: $LEADER_IPFS_CLUSTER_MULTIADDR"
echo "Using IPFS cluster path: $IPFS_CLUSTER_PATH"

if [ -n "$DOCKER_DEBUG" ]; then
   set -x
fi

if [ "$IS_LEADER" = "true" ]; then
    echo "This is the leader node. No need to bootstrap."
else
    # Make sure leader ipfs cluster multiaddr is set
    if [ -z "$LEADER_IPFS_CLUSTER_MULTIADDR" ]; then
        echo "LEADER_IPFS_CLUSTER_MULTIADDR is not set. Exiting..."
        exit 1
    fi
fi

if [ -z "$CLUSTER_SECRET" ]; then
    echo "CLUSTER_SECRET is not set. Exiting..."
    exit 1
fi

# Function to run command as ipfs user
run_as_ipfs() {
    if [ $(id -u) -eq 0 ]; then
        su -p "$user" -c "$*"
    else
        "$@"
    fi
}

# watch the restart file and exit if it exists
watch_restart_file() {
    while true; do
        if [ -f "$RESTART_FILE" ]; then
            echo "Restart file detected. Removing file and exiting..."
            rm "$RESTART_FILE"
            kill -TERM $1
            exit 0
        fi
        sleep $CHECK_INTERVAL
    done
}

fetch_trusted_peers() {
    echo "Fetching trusted peers from listener service..."
    trusted_peers=$(curl -s ${LISTENER_URL}/trusted-peers)
    if [ $? -ne 0 ]; then
        echo "Failed to fetch trusted peers. Exiting..."
        exit 1
    fi
    echo "Fetched trusted peers: $trusted_peers"
}

update_service_json() {
    echo "Updating service.json with trusted peers and leader address..."
    FOLLOWER_SERVICE_JSON=${IPFS_CLUSTER_PATH}/$CLUSTER_NAME/service.json
    jq --argjson peers "$trusted_peers" \
       --arg leader "$LEADER_IPFS_CLUSTER_MULTIADDR" \
       '.consensus.crdt.trusted_peers = $peers | .cluster.peer_addresses = [$leader]' \
       ${FOLLOWER_SERVICE_JSON} > ${FOLLOWER_SERVICE_JSON}.tmp
    mv ${FOLLOWER_SERVICE_JSON}.tmp ${FOLLOWER_SERVICE_JSON}
    echo "Updated service.json"
}

start_ipfs_cluster() {
    # Check if we're root
    if [ $(id -u) -eq 0 ]; then
        echo "Changing user to $user"
        # ensure directories are writable
        if ! run_as_ipfs test -w "${IPFS_CLUSTER_PATH}"; then
            chown -R -- "$user" "${IPFS_CLUSTER_PATH}"
        fi
        
        # Re-execute the script as the ipfs user
        exec su -p "$user" -c "$0 $*"
    fi

    # Only ipfs user can get here
    ipfs-cluster-follow --version

    if [ -e "${IPFS_CLUSTER_PATH}/service.json" ]; then
        echo "Found IPFS cluster configuration at ${IPFS_CLUSTER_PATH}"
    else
        echo "Initializing default configuration..."
        ipfs-cluster-follow "$CLUSTER_NAME" init https://example.com
        ipfs-cluster-service init --consensus "${IPFS_CLUSTER_CONSENSUS}"
        # copy service.json to $CLUSTER_NAME config directory
        cp ${IPFS_CLUSTER_PATH}/service.json ${IPFS_CLUSTER_PATH}/$CLUSTER_NAME/service.json
    fi

    # Fetch trusted peers and update service.json for follower config for $CLUSTER_NAME
    fetch_trusted_peers
    update_service_json

    echo "Starting IPFS Cluster Follower"
    echo "Watching for restart file: $RESTART_FILE"
    echo "Check interval: ${CHECK_INTERVAL} second(s)"

    # Start the restart file watcher in the background
    watch_restart_file $$ &

    exec ipfs-cluster-follow "$CLUSTER_NAME" run "$@"
}

# Start the IPFS cluster
start_ipfs_cluster "$@"