# Default Miner

The default miner simply splits the incoming document into individual sentences and then forms each chunk by concatenating adjacent sentences until the token limit specified by the validator is reached. This is not an optimal strategy and will likely result in very low yields or deregistration.

For tips on how build a better miner, [view our guide](./miner_guide.md).

## Prerequisites

- Review the minimum computational requirements for the desired role

- Ensure you have gone through the [checklist for validating and mining](https://docs.bittensor.com/subnets/checklist-for-validating-mining)

- Ensure that you have registered a hotkey for our subnet

- This repository requires python3.8 or higher

## Installation/Setup

This repository requires python 3.8 or higher. The following command will install the necessary dependencies and clone the repository:

```bash
curl -sSL https://raw.githubusercontent.com/VectorChat/chunking_subnet/main/setup.sh | bash
```

It is highly recomended that you write your own logic for neurons/miner.py:forward to optimize chunking and maximize rewards. For guidance creating a custom miner, check out [the miner guide](./miner_guide.md).

To run the miner, issue the following command:

```bash
bash run-miner.sh
```

Make sure to have your environment variables properly set in your `.env` file.

To run the miner with relay mining protection (`lucid`), follow the setup instructions [here](./lucid/setup.md) for miners.

> [!WARNING]
> You will need to add `OPENAI_API_KEY` to your `.env` file if you allow for checking of fuzzy/semi duplicate requests.

## Configuration

| Argument                                | Description                                                                                                                     |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `--blacklist.force_validator_permit`    | If set, force incoming requests to have a validator permit.                                                                     |
| `--blacklist.allow_non_registered`      | If set, will accept queries from non registered entities. (Dangerous!)                                                          |
| `--blacklist.minimum_stake`             | If set, force incoming requests to have a weight-settable stake.                                                                |
| `--neuron.disable_verification`         | If set, will accept queries without verifying (nonce). (Dangerous!)                                                             |
| `--neuron.synapse_verify_allowed_delta` | The allowed delta for synapse verification in nanoseconds.                                                                      |
| `--neuron.relay_embed_threshold`        | The threshold of cosine similarity to use when comparing two request documents.                                                 |
| `--neuron.no_check_ipfs`                | If set, does not run IPFS/relay mining related checks.                                                                          |
| `--neuron.no_check_duplicate_ipfs`      | If set, does not check for exact or fuzzy duplicate requests in IPFS.                                                           |
| `--neuron.no_serve`                     | If set, skips serving the miner axon on chain                                                                                   |
| `--neuron.reconnect.min_seconds`        | The minimum number of seconds to wait before reconnecting to the network.                                                       |
| `--neuron.reconnect.max_seconds`        | The maximum number of seconds to wait before reconnecting to the network (makes sure exponential backoff is not too aggressive) |
| `--neuron.reconnect.max_attempts`       | The maximum number of attempts to reconnect to the network.                                                                     |
