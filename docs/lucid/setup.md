# Setup

The components for this relay mining update currently run independently of the validator and miner scripts. For validators there is [`compose-validator.yml`](../compose-validator.yml) and for miners there is [`compose-miner.yml`](../compose-miner.yml).

## Both Validators and Miners

First, make sure you have Docker installed and setup.

If you are on Windows, please follow [the official docs](https://docs.docker.com/desktop/install/windows-install/).

For Mac, please follow [these docs](https://docs.docker.com/desktop/install/mac-install/). 

For Linux, depending on your distro, you can follow either the [official docs](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) or [these docs](https://docs.sevenbridges.com/docs/install-docker-on-linux)

You can also try typing `docker` into your terminal and it may recommend a package manager to install docker.

For mainnet, make sure the following is in your `.env` file:

```txt
...
# common
CLUSTER_SECRET=""
LEADER_IPFS_MULTIADDR=""
LEADER_IPFS_CLUSTER_MULTIADDR=""
LEADER_IPFS_CLUSTER_ID=""
LISTENER_ARGS="--netuid 40 --min-stake 1000"
BT_DIR="/root/.bittensor"

# validator-only
INSCRIBER_ARGS="--netuid 40 --bittensor-coldkey-name YOUR_COLDKEY --bittensor-hotkey-name YOUR_HOTKEY"
...
```

For testnet, use a separate env file, something like `.env.testnet`.

```txt
...
# testnet
CLUSTER_SECRET=""
SWARM_SECRET=""
LEADER_IPFS_MULTIADDR=""
LEADER_IPFS_CLUSTER_MULTIADDR=""
LEADER_IPFS_CLUSTER_ID=""
LISTENER_ARGS="--netuid 166 --min-stake 1000 --ws-url wss://test.finney.opentensor.ai:443/"
BT_DIR="/root/.bittensor"

# validator-only
INSCRIBER_ARGS="--netuid 166 --bittensor-coldkey-name YOUR_COLDKEY --bittensor-hotkey-name YOUR_HOTKEY --ws-url wss://test.finney.opentensor.ai:443/"
...
```

This can then be specified when running the docker compose file, like so:

```bash
docker compose -f compose-validator.yml --env-file .env.testnet up --build -d
```

Description of environment variables:

| Variable                        | Description                                                                                                     |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| CLUSTER_SECRET                  | The secret used to encrypt the IPFS Cluster.                                                                    |
| IPFS_SWARM_KEY                  | The key used to encrypt the IPFS Swarm.                                                                         |
| LEADER_IPFS_MULTIADDR           | The multiaddress of the IPFS leader.                                                                            |
| LEADER_IPFS_CLUSTER_MULTIADDR   | The multiaddress of the IPFS Cluster leader.                                                                    |
| LEADER_IPFS_CLUSTER_ID          | The ID of the IPFS Cluster leader.                                                                              |
| LISTENER_ARGS                   | The arguments passed to the listener service.                                                                   |
| INSCRIBER_ARGS (validator-only) | The arguments passed to the inscriber service. Make sure to fill out your correct bittensor coldkey and hotkey. |

### Firewall Setup

Make sure to expose the swarm ports for both IPFS and IPFS Cluster.

| Service      | Port |
| ------------ | ---- |
| IPFS         | 4001 |
| IPFS Cluster | 9096 |

Here's a snippet to set that up if you use ufw:

```bash
# Allow IPFS Swarm port
sudo ufw allow 4001/tcp

# Allow IPFS Cluster Swarm port
sudo ufw allow 9096/tcp

# Enable the firewall if it's not already active
sudo ufw enable

Everything else can be walled off as normal (other than your axon port(s))
```

## Validator

To run, use the following command:

```bash
docker compose -f compose-validator.yml up --build -d
```

To view logs, use:

```bash
docker compose -f compose-validator.yml logs -f -n=250
```

## Miner

> [!NOTE]
> If you run multiple miners on the same machine, you only need to setup these services once. They can each use the same IPFS node and IPFS Cluster Follower node to validate and list pins.

To run, use the following command:

```bash
docker compose -f compose-miner.yml up --build -d
```

To view logs, use:

```bash
docker compose -f compose-miner.yml logs -f -n=250
```

To stop, use:

```bash
docker compose -f compose-miner.yml down
```

## Useful Docker Commands

```bash
# check existing docker process
docker ps

# check logs of compose file
docker compose -f FILE logs -f -n=NUMLINES

# get into a container
docker exec -it NAME sh

# view live stats
docker stats
```
