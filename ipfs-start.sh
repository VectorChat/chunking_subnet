#!/bin/sh
set -e

user=ipfs
repo="$IPFS_PATH"

if [ "$(id -u)" -eq 0 ]; then
  echo "Changing user to $user"
  # ensure folder is writable
  gosu "$user" test -w "$repo" || chown -R -- "$user" "$repo"
  # restart script with new privileges
  exec gosu "$user" "$0" "$@"
fi

# 2nd invocation with regular user
ipfs version

if [ -e "$repo/config" ]; then
  echo "Found IPFS fs-repo at $repo"
else
  ipfs init ${IPFS_PROFILE:+"--profile=$IPFS_PROFILE"}
  ipfs config Addresses.API /ip4/0.0.0.0/tcp/5001
  ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8080

  # Set up the swarm key
  SWARM_KEY_FILE="$repo/swarm.key"
  SWARM_KEY_PERM=0400

  chmod $SWARM_KEY_PERM "$SWARM_KEY_FILE"
  echo "Swarm key set up at $SWARM_KEY_FILE"

  # Additional configuration for private network
  ipfs bootstrap rm --all
  echo "Removed all bootstrap nodes"
  ipfs config --json Discovery.MDNS.Enabled false
  echo "Disabled mDNS discovery"
  ipfs config --json Routing.Type '"dht"'
  echo "Set routing type to DHT"

  if [ "$IS_LEADER" = "true" ]; then
    echo "This is the leader node. No need to bootstrap."
  else
    if [ -n "$LEADER_IPFS_MULTIADDR" ]; then
      ipfs bootstrap add "$LEADER_IPFS_MULTIADDR"
      echo "Added leader IPFS multiaddress: $LEADER_IPFS_MULTIADDR"
    else
      echo "No leader IPFS multiaddress specified."
      exit 1
    fi
  fi
fi

find /container-init.d -maxdepth 1 -type f -iname '*.sh' -print0 | sort -z | xargs -n 1 -0 -r container_init_run

exec ipfs daemon --migrate=true --agent-version-suffix=docker "$@"