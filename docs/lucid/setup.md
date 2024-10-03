# Lucid (Relay Mining Prevention) Setup

## Prerequsities

Follow the [lucid setup guide](https://github.com/VectorChat/lucid/blob/main/docs/setup.md)

> [!NOTE]
> The below steps assume you are in the `lucid` repo.

## Testnet

For testnet, make sure the following is in you `.env.testnet` file

```txt
CLUSTER_SECRET="902a6de47a797c7b1a281d64f060e23a1bda8483aef1727c44bb579ea9db04fd"
SWARM_SECRET="3cfb7dc08eed0a11ef5d0a040e1cd376956515abb5861f750f9c99a05d0f599b"
LEADER_IPFS_MULTIADDR="/ip4/104.248.48.55/tcp/4001/p2p/12D3KooWBSh847xmDVpY1vdQWJXYiNE4jj1UPP8FZXmCZ6hgczkx"
LEADER_IPFS_CLUSTER_MULTIADDR="/ip4/104.248.48.55/tcp/9096/p2p/12D3KooWRVhcSotB5H32PEydK5t5L1zbzxsCaWcFgZEbJFzU2YRe"
LEADER_IPFS_CLUSTER_ID="12D3KooWRVhcSotB5H32PEydK5t5L1zbzxsCaWcFgZEbJFzU2YRe"
LISTENER_ARGS="--netuid 166 --min-stake 0.5 --ws-url wss://test.finney.opentensor.ai:443/"
BT_DIR="/root/.bittensor"

# validator-only
INSCRIBER_ARGS="--netuid 166 --bittensor-coldkey-name test_chunking_owner --bittensor-hotkey-name validator-1 --ws-url wss://test.finney.opentensor.ai:443/"
```

To run components for a validator, use:

```bash
docker compose --env-file .env.testnet -f compose-validator.yml up --build -d
```

To run components for a miner, use:

```bash
docker compose --env-file .env.testnet -f compose-miner.yml up --build -d
```

## Mainnet

TBD
