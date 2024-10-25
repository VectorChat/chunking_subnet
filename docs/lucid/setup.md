# Lucid (Relay Mining Prevention) Setup

Clone the `lucid` repo and navigate into the directory.

```bash
cd .. # navigate to the root directory or any directory above the `chunking_subnet` repo
git clone https://github.com/VectorChat/lucid.git
cd lucid
```

Setup testnet and/or mainnet environment variables as needed

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

## Mainnet

For mainnet, make sure the following is in your `.env` file

```txt
...
# common
CLUSTER_SECRET="4e9e8c3de87dd495dc9ce6304ee7006a8228f64bad53f0870576adb06207de55"
SWARM_SECRET="261cacbfcf697faa54a7196b824e461adab7c188236ac79ee836e5b7c87f7d9a"
LEADER_IPFS_MULTIADDR="/ip4/137.184.138.42/tcp/4001/p2p/12D3KooWPbLoA76oDwYwmuiXW6JFd7Bvbh4gvDfKzNjMWesB1JqV"
LEADER_IPFS_CLUSTER_MULTIADDR="/ip4/137.184.138.42/tcp/9096/p2p/12D3KooWKDk3TGgkKodAjo6ydRZjtD6TeZT1dvqvb6Bm2h8RHRMU"
LEADER_IPFS_CLUSTER_ID="12D3KooWKDk3TGgkKodAjo6ydRZjtD6TeZT1dvqvb6Bm2h8RHRMU"
LISTENER_ARGS="--netuid 40 --min-stake 10000 --ws-url ws://mainnet-lite:9944 --log-level info"
BT_DIR="/root/.bittensor"

# validator-only
INSCRIBER_ARGS="--netuid 40 --bittensor-coldkey-name <YOUR_COLDKEY> --bittensor-hotkey-name <YOUR_HOTKEY> --ws-url ws://mainnet-lite:9944 --log-level info"
...
```

> [!NOTE]
> Replace `<YOUR_COLDKEY>` and `<YOUR_HOTKEY>` with your coldkey and hotkey names.

Finally, the rest of the setup and startup commands can be found in the [lucid setup guide](https://github.com/VectorChat/lucid/blob/main/docs/setup.md).
