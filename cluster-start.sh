#!/bin/sh

set -e
user=ipfs

CHECK_INTERVAL=${CHECK_INTERVAL:-1}

RESTART_FILE=${RESTART_FILE:-/data/ipfs-cluster/restart}

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

# Function to run command as ipfs user
run_as_ipfs() {
    if [ $(id -u) -eq 0 ]; then
        su -p "$user" -c "$*"
    else
        "$@"
    fi
}

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
    ipfs-cluster-service --version

    if [ -e "${IPFS_CLUSTER_PATH}/service.json" ]; then
        echo "Found IPFS cluster configuration at ${IPFS_CLUSTER_PATH}"
    else
        echo "This container only runs ipfs-cluster-service. ipfs needs to be run separately!"
        echo "Initializing default configuration..."
        ipfs-cluster-service init --consensus "${IPFS_CLUSTER_CONSENSUS}"
    fi

    echo "Starting IPFS Cluster"
    echo "Watching for restart file: $RESTART_FILE"
    echo "Check interval: ${CHECK_INTERVAL} second(s)"

    # Start the restart file watcher in the background
    watch_restart_file $$ &

    # Start ipfs-cluster-service in the foreground
    exec ipfs-cluster-service daemon --bootstrap "$LEADER_IPFS_CLUSTER_MULTIADDR" "$@"
}

# Start the IPFS cluster
start_ipfs_cluster "$@"